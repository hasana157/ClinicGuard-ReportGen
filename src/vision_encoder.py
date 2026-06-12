"""
Vision encoder module using torchxrayvision.

Loads a pre-trained DenseNet121 model trained on CheXpert/MIMIC-CXR and defines a
classification head for pathology prediction. Provides methods for spatial feature
extraction and PIL image preprocessing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
from src.config import (
    VisionEncoderConfig, ProjectConfig, PATHOLOGY_LABELS, TORCHXRAYVISION_LABELS, 
    TXV_TO_PRIMARY, get_config
)


class GroundedVisionEncoder(nn.Module):
    """
    DenseNet121 vision encoder leveraging torchxrayvision pre-trained weights
    and a custom classification head for primary pathologies.
    """

    def __init__(self, config: Optional[VisionEncoderConfig] = None):
        """
        Initialize the vision encoder.

        Args:
            config: Configuration for the vision encoder.
        """
        super().__init__()
        self.config = config if config else VisionEncoderConfig()
        
        # Load torchxrayvision model
        # Try importing torchxrayvision. If not present, we will define a mock backbone
        # to ensure the code executes cleanly in any environment.
        try:
            import torchxrayvision as xrv
            print(f"Loading torchxrayvision model: {self.config.model_name}...")
            # Load pre-trained DenseNet
            self.backbone = xrv.models.DenseNet(weights=self.config.model_name)
            self.has_xrv = True
        except Exception as e:
            print(f"torchxrayvision not found or failed to load: {e}. Defining standard DenseNet121.")
            import torchvision.models as models
            # Use torchvision densenet121 as fallback
            self.backbone = models.densenet121(pretrained=True)
            self.has_xrv = False

        # Custom classification head mapping global features (1024) to our primary pathology labels
        # DensetNet121 bottleneck is 1024
        self.feature_dim = 1024
        self.dropout = nn.Dropout(p=self.config.dropout_rate)
        self.classifier = nn.Linear(self.feature_dim, len(PATHOLOGY_LABELS))
        
        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
    def get_target_layer(self) -> nn.Module:
        """
        Get the target layer for Grad-CAM grounding (usually the last dense block).

        Returns:
            The PyTorch nn.Module target layer.
        """
        if self.has_xrv:
            # torchxrayvision DenseNet uses self.backbone.features
            # Under the hood, this is a standard DenseNet features block
            return self.backbone.features.norm5
        else:
            return self.backbone.features.denseblock4.denselayer16

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image tensor of shape (B, 3, 224, 224).

        Returns:
            Dict containing logits, probabilities, global features, and spatial features.
        """
        # Feature extraction
        if self.has_xrv:
            # torchxrayvision DenseNet forward features
            # Returns a feature map of shape (B, 1024, 7, 7)
            features = self.backbone.features(x)
        else:
            features = self.backbone.features(x)
            
        spatial_features = features  # (B, 1024, 7, 7)
        
        # Global pooling
        global_features = F.adaptive_avg_pool2d(spatial_features, (1, 1)).squeeze(-1).squeeze(-1) # (B, 1024)
        
        # Classification
        x_dropout = self.dropout(global_features)
        logits = self.classifier(x_dropout)  # (B, 14)
        probabilities = torch.sigmoid(logits)  # (B, 14)
        
        return {
            "logits": logits,
            "probabilities": probabilities,
            "features": global_features,
            "spatial_features": spatial_features,
        }

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract global and spatial features only.

        Args:
            x: Input image tensor of shape (B, 1, 224, 224).

        Returns:
            Tuple of (global_features, spatial_features).
        """
        out = self.forward(x)
        return out["features"], out["spatial_features"]

    def classify(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Classify chest X-ray and return labels and confidences.

        Args:
            x: Input image tensor (B, 3, 224, 224).
            threshold: Confidence threshold for positive findings.

        Returns:
            Dict with labels, confidences, and predictions.
        """
        out = self.forward(x)
        probabilities = out["probabilities"][0].detach().cpu().numpy()
        
        predictions = (probabilities >= threshold).astype(float)
        
        detected_findings = []
        confidences = {}
        for idx, label in enumerate(PATHOLOGY_LABELS):
            conf = float(probabilities[idx])
            confidences[label] = conf
            if conf >= threshold:
                detected_findings.append(label)
                
        return {
            "detected_findings": detected_findings,
            "confidences": confidences,
            "predictions": predictions,
            "raw_probabilities": probabilities,
        }


# ============================================================================
# Preprocessing Helper for Model Inference
# ============================================================================

def preprocess_for_model(image: Image.Image, size: int = 224) -> torch.Tensor:
    """
    Preprocess a PIL image for input to the GroundedVisionEncoder.

    Args:
        image: PIL Image object.
        size: Target image dimensions.

    Returns:
        Tensor of shape (1, 3, size, size) ready for model forward pass (3 channels for DenseNet).
    """
    # Convert image to grayscale (single channel)
    image = image.convert("L")
    # Resize
    image = image.resize((size, size))
    image_np = np.array(image, dtype=np.float32)
    
    # Scale to [0, 1]
    image_np = image_np / 255.0
    
    # Rescale to [-1024, 1024] expected by torchxrayvision models
    image_np = (image_np * 2048.0) - 1024.0
    
    # Convert grayscale to 3-channel by repeating (DenseNet expects RGB)
    image_np = np.stack([image_np, image_np, image_np], axis=0)  # (3, size, size)
    
    # Convert to tensor and add batch dimension: (1, 3, size, size)
    tensor = torch.from_numpy(image_np).unsqueeze(0).float()
    return tensor


def load_pretrained_encoder(config: Optional[ProjectConfig] = None) -> GroundedVisionEncoder:
    """
    Factory function to load the pre-trained encoder.

    Args:
        config: Project Master configuration.

    Returns:
        GroundedVisionEncoder instance.
    """
    cfg = config if config else get_config()
    model = GroundedVisionEncoder(config=cfg.vision)
    return model


if __name__ == "__main__":
    # Test encoder forward pass
    encoder = GroundedVisionEncoder()
    mock_input = torch.randn(2, 3, 224, 224)
    out = encoder(mock_input)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Probabilities shape: {out['probabilities'].shape}")
    print(f"Global features shape: {out['features'].shape}")
    print(f"Spatial features shape: {out['spatial_features'].shape}")
    
    # Test preprocessing
    dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 300), dtype=np.uint8))
    img_t = preprocess_for_model(dummy_img)
    print(f"Preprocessed shape: {img_t.shape}")
    print(f"Preprocessed range: [{img_t.min():.1f}, {img_t.max():.1f}]")
