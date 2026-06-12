"""
Uncertainty quantification module.

Implements Monte Carlo dropout inference for uncertainty estimation on deep learning model
outputs, confidence calibration, and confidence-based refusal mechanisms.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from src.config import get_config


class UncertaintyQuantifier:
    """Uncertainty quantification and calibration engine."""

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int = 10,
        temperature: float = 1.0
    ):
        """
        Initialize the quantifier.

        Args:
            model: PyTorch model.
            num_mc_samples: Number of forward passes with dropout enabled.
            temperature: Calibration temperature parameter.
        """
        self.model = model
        self.num_mc_samples = num_mc_samples
        self.temperature = temperature
        self.config = get_config()

    def enable_dropout(self):
        """Enables dropout layers during inference for Monte Carlo sampling."""
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def mc_dropout_inference(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Perform Monte Carlo dropout inference to estimate model prediction uncertainty.

        Args:
            image_tensor: Input image tensor (1, 1, 224, 224).

        Returns:
            Dict containing mean probability, standard deviation, and prediction entropy.
        """
        self.model.eval()
        self.enable_dropout()  # Enable dropout layers only
        
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        probabilities_list = []
        
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                out = self.model(image_tensor)
                # Calibrate probabilities with temperature scaling before sigmoid
                logits_scaled = out["logits"] / self.temperature
                probs = torch.sigmoid(logits_scaled)
                probabilities_list.append(probs.cpu().numpy())
                
        # Stack probabilities shape: (num_samples, batch_size, num_pathologies)
        stacked = np.stack(probabilities_list, axis=0)  
        
        # Calculate statistics across MC passes
        mean_probs = np.mean(stacked, axis=0)[0]  # Shape: (num_pathologies,)
        std_probs = np.std(stacked, axis=0)[0]    # Standard deviation (uncertainty)
        
        # Calculate entropy: -p*log(p) - (1-p)*log(1-p)
        epsilon = 1e-8
        entropy = -(mean_probs * np.log(mean_probs + epsilon) + 
                    (1.0 - mean_probs) * np.log(1.0 - mean_probs + epsilon))
                    
        return {
            "mean": mean_probs,
            "std": std_probs,
            "entropy": entropy
        }

    def get_uncertainty_scores(self, image_tensor: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        Get uncertainty statistics for all pathologies.

        Args:
            image_tensor: Input image.

        Returns:
            Dict mapping pathology to {confidence, std, entropy}.
        """
        stats = self.mc_dropout_inference(image_tensor)
        
        from src.config import PATHOLOGY_LABELS
        result = {}
        for idx, label in enumerate(PATHOLOGY_LABELS):
            result[label] = {
                "confidence": float(stats["mean"][idx]),
                "std": float(stats["std"][idx]),
                "entropy": float(stats["entropy"][idx])
            }
        return result

    def should_refuse(self, label: str, stats: Dict[str, float], threshold: float = 0.15) -> bool:
        """
        Determine if the prediction for a label should be refused due to excessive uncertainty.

        Args:
            label: Pathology label.
            stats: Dict with 'std' and 'confidence'.
            threshold: Standard deviation threshold.

        Returns:
            True if prediction is too uncertain and should be refused.
        """
        # Refuse if the standard deviation (model disagreement) is above the threshold
        # or if the confidence is borderline (0.4 to 0.6) and standard deviation is moderate.
        std = stats.get("std", 0.0)
        confidence = stats.get("confidence", 0.0)
        
        if std >= threshold:
            return True
        if 0.45 <= confidence <= 0.55 and std > 0.08:
            return True
            
        return False


if __name__ == "__main__":
    # Test MC dropout
    from src.vision_encoder import GroundedVisionEncoder
    model = GroundedVisionEncoder()
    quantifier = UncertaintyQuantifier(model, num_mc_samples=5)
    
    dummy_input = torch.randn(1, 1, 224, 224)
    scores = quantifier.get_uncertainty_scores(dummy_input)
    
    for label, metrics in list(scores.items())[:3]:
        print(f"Pathology: {label}")
        print(f"  Confidence: {metrics['confidence']:.2f}")
        print(f"  Uncertainty (std): {metrics['std']:.2f}")
        print(f"  Entropy: {metrics['entropy']:.2f}")
        print(f"  Should refuse: {quantifier.should_refuse(label, metrics)}")
