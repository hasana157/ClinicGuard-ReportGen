"""
Data loader and dataset definition for the IU X-Ray dataset.

This module loads the Indiana University Chest X-Ray dataset (IU X-Ray) from HuggingFace,
handles train/val/test splits, parses report text to extract pathology labels, and
provides PyTorch DataLoader instances.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
from src.config import PATHOLOGY_LABELS, ProjectConfig

# Synonym dictionary for pathology keyword extraction
PATHOLOGY_SYNONYMS = {
    "Atelectasis": ["atelectasis", "atelectases", "subsegmental atelectasis", "platelike atelectasis", "discoid atelectasis"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement", "enlarged cardiac silhouette", "prominent cardiac silhouette"],
    "Consolidation": ["consolidation", "consolidations", "airspace consolidation", "focal consolidation"],
    "Edema": ["edema", "pulmonary edema", "vascular congestion", "fluid overload"],
    "Effusion": ["effusion", "effusions", "pleural effusion", "pleural effusions", "fluid in the pleural space"],
    "Emphysema": ["emphysema", "emphysematous changes", "hyperinflation"],
    "Fibrosis": ["fibrosis", "scarring", "fibrotic", "interstitial scarring"],
    "Hernia": ["hernia", "hiatal hernia"],
    "Infiltration": ["infiltration", "infiltrations", "infiltrates", "infiltrate", "interstitial infiltrate", "airspace disease"],
    "Mass": ["mass", "masses", "large nodule", "tumor"],
    "Nodule": ["nodule", "nodules", "nodular opacity", "nodular opacities"],
    "Pleural_Thickening": ["pleural thickening", "pleural thickening/scarring", "apical thickening"],
    "Pneumonia": ["pneumonia", "pneumonias", "infectious process", "consolidation suggesting pneumonia"],
    "Pneumothorax": ["pneumothorax", "pneumothoraces", "collapsed lung"],
}

class IUXRayDataset(Dataset):
    """Indiana University Chest X-Ray Dataset loader."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Any] = None,
        config: Optional[ProjectConfig] = None,
    ):
        """
        Initialize the dataset.

        Args:
            split: One of 'train', 'val', or 'test'.
            transform: Albumentations or Torchvision transformations.
            config: Master project configuration.
        """
        self.split = split
        self.transform = transform
        self.config = config if config else ProjectConfig()
        
        # Load HuggingFace dataset
        # In a real environment, we'd use:
        # from datasets import load_dataset
        # self.hf_dataset = load_dataset(self.config.vision.model_name)
        # For simplicity and robustness during test/offline runs, we'll download/simulate 
        # or load from the cache. Let's create a robust fallback mechanism.
        self.samples = []
        self._load_dataset()

    def _load_dataset(self):
        """Loads and parses the dataset samples."""
        # We will attempt to load the datasets library. If not installed, we fallback to a demo set.
        try:
            from datasets import load_dataset
            # Using 'dz-osamu/IU-Xray' as it contains pre-paired images and reports
            print(f"Loading {self.config.vision.model_name} from HuggingFace...")
            dataset = load_dataset("dz-osamu/IU-Xray", split="train")
            
            # Let's perform deterministic split
            np.random.seed(self.config.training.seed)
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)
            
            n_total = len(dataset)
            n_train = int(n_total * self.config.training.seed) # using seed or ratio
            # Let's use config ratios
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)
            
            if self.split == "train":
                split_indices = indices[:n_train]
            elif self.split == "val":
                split_indices = indices[n_train:n_train + n_val]
            else:
                split_indices = indices[n_train + n_val:]
                
            for idx in split_indices:
                item = dataset[int(idx)]
                # Extract image and reports
                image = item.get("image")
                findings = item.get("findings", "")
                impression = item.get("impression", "")
                
                # Combine reports
                report_text = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
                labels = extract_labels_from_report(report_text, PATHOLOGY_LABELS)
                
                self.samples.append({
                    "image": image,
                    "report": report_text,
                    "findings": findings,
                    "impression": impression,
                    "labels": labels,
                    "id": f"IU-{idx}",
                })
        except Exception as e:
            print(f"HuggingFace dataset load failed or offline: {e}. Generating mock dataset for execution.")
            self._generate_mock_dataset()

    def _generate_mock_dataset(self):
        """Generates a mock dataset for testing and running without internet access."""
        np.random.seed(self.config.training.seed + (1 if self.split == "val" else 2 if self.split == "test" else 0))
        num_samples = 40 if self.split == "train" else 10
        
        # Standard mock findings
        mock_findings = [
            "The heart size is normal. The lungs are clear. No pleural effusion or pneumothorax is seen.",
            "Cardiomegaly is present. Prominence of the pulmonary vasculature. Pleural effusions are noted bilaterally.",
            "Focal consolidation in the right lower lobe, suspicious for pneumonia. No pneumothorax.",
            "Mild atelectasis is noted at the lung bases. Otherwise normal study.",
            "Lungs are hyperinflated with flattening of diaphragms, consistent with emphysema. No acute infiltrate.",
        ]
        
        mock_impressions = [
            "No acute cardiopulmonary disease.",
            "Cardiomegaly and bilateral pleural effusions.",
            "Right lower lobe pneumonia.",
            "Mild basilar atelectasis.",
            "Emphysema without acute process.",
        ]
        
        for i in range(num_samples):
            # Create a mock grayscale image (224x224)
            img_arr = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            img = Image.fromarray(img_arr)
            
            idx = np.random.choice(len(mock_findings))
            findings = mock_findings[idx]
            impression = mock_impressions[idx]
            report_text = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
            labels = extract_labels_from_report(report_text, PATHOLOGY_LABELS)
            
            self.samples.append({
                "image": img,
                "report": report_text,
                "findings": findings,
                "impression": impression,
                "labels": labels,
                "id": f"MOCK-{self.split}-{i}",
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = sample["image"]
        
        # Apply transformation
        if self.transform:
            # Albumentations expects numpy, torchvision expects PIL/Tensor
            # Let's check transform type
            try:
                # Assuming torchvision transform
                image_tensor = self.transform(image)
            except Exception:
                # Fallback: Convert to numpy and transform
                img_np = np.array(image)
                image_tensor = torch.tensor(img_np).float().unsqueeze(0) / 255.0
        else:
            # Standard conversion
            img_np = np.array(image.convert("L").resize((224, 224)))
            image_tensor = torch.tensor(img_np).float().unsqueeze(0) / 255.0
            
        return {
            "image": image_tensor,
            "report": sample["report"],
            "findings": sample["findings"],
            "impression": sample["impression"],
            "labels": torch.tensor(sample["labels"], dtype=torch.float32),
            "metadata": {"id": sample["id"]},
        }


class MIMICCXRLoader(Dataset):
    """Stub loader for MIMIC-CXR dataset, demonstrating credential requirements."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Access to MIMIC-CXR requires PhysioNet credentials, CITI human subjects training, "
            "and signed Data Use Agreement (DUA). Please download MIMIC-CXR-JPG manually and "
            "use a custom script to parse it."
        )


class PadChestLoader(Dataset):
    """Stub loader for PadChest dataset, demonstrating request requirements."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Access to PadChest requires BIMCV request approval. "
            "Please register at BIMCV website to obtain the dataset."
        )


# ============================================================================
# Helper Functions
# ============================================================================

def extract_labels_from_report(report_text: str, pathology_labels: List[str]) -> List[float]:
    """
    Extract labels from report text using keyword matching and negation detection.

    Args:
        report_text: Raw radiology report text.
        pathology_labels: List of labels to extract.

    Returns:
        List of floats: 1.0 (positive), 0.0 (negative/not mentioned), -1.0 (uncertain).
    """
    report_lower = report_text.lower()
    labels = [0.0] * len(pathology_labels)
    
    # Negation rules: check if finding is preceded by no, normal, clear, negative, free of, etc.
    negation_patterns = [
        r"\bno\s+evidence\s+of\s+{finding}",
        r"\bno\s+{finding}",
        r"\bwithout\s+{finding}",
        r"\bfree\s+of\s+{finding}",
        r"\bnegative\s+for\s+{finding}",
        r"\bnormal\s+{finding}",
        r"\bclear\s+of\s+{finding}",
        r"\bnot\s+seen\b",
        r"\babsent\b",
        r"\bruled\s+out\b"
    ]
    
    # Uncertainty rules: check if finding is associated with queries, may represent, borderlines
    uncertain_patterns = [
        r"\brule\s+out\s+{finding}",
        r"\bpossible\s+{finding}",
        r"\bpossibly\s+{finding}",
        r"\bsuggestive\s+of\s+{finding}",
        r"\bmay\s+represent\s+{finding}",
        r"\bcannot\s+exclude\s+{finding}",
        r"\bborderline\s+{finding}",
        r"\bcompatible\s+with\s+{finding}",
        r"\bquery\s+{finding}"
    ]

    for idx, pathology in enumerate(pathology_labels):
        synonyms = PATHOLOGY_SYNONYMS.get(pathology, [pathology.lower()])
        
        # Check presence
        present = False
        uncertain = False
        negated = False
        
        for syn in synonyms:
            syn_escaped = re.escape(syn)
            if syn in report_lower:
                # Check for negation first
                for pattern in negation_patterns:
                    # check pattern matching around the finding
                    filled_pattern = pattern.format(finding=syn_escaped)
                    if re.search(filled_pattern, report_lower):
                        negated = True
                        break
                        
                # If not negated, check for uncertainty
                if not negated:
                    for pattern in uncertain_patterns:
                        filled_pattern = pattern.format(finding=syn_escaped)
                        if re.search(filled_pattern, report_lower):
                            uncertain = True
                            break
                
                if not negated:
                    present = True
                    break
        
        if present:
            labels[idx] = -1.0 if uncertain else 1.0
        else:
            labels[idx] = 0.0
            
    return labels


def get_report_sections(report_text: str) -> Dict[str, str]:
    """
    Split a radiology report into standard sections.

    Args:
        report_text: Raw report text.

    Returns:
        Dict containing FINDINGS and IMPRESSION sections.
    """
    sections = {"INDICATION": "", "FINDINGS": "", "IMPRESSION": ""}
    
    # Simple regex parsing
    findings_match = re.search(r"FINDINGS:(.*?)(?:IMPRESSION:|$)", report_text, re.DOTALL | re.IGNORECASE)
    impression_match = re.search(r"IMPRESSION:(.*?)$", report_text, re.DOTALL | re.IGNORECASE)
    
    if findings_match:
        sections["FINDINGS"] = findings_match.group(1).strip()
    if impression_match:
        sections["IMPRESSION"] = impression_match.group(1).strip()
        
    return sections


def get_dataloaders(
    config: ProjectConfig,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get DataLoader instances for train, val, and test splits.

    Args:
        config: Mastering configuration.
        train_transform: Preprocessing transforms for train set.
        eval_transform: Preprocessing transforms for val/test set.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_set = IUXRayDataset(split="train", transform=train_transform, config=config)
    val_set = IUXRayDataset(split="val", transform=eval_transform, config=config)
    test_set = IUXRayDataset(split="test", transform=eval_transform, config=config)
    
    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,  # 0 is safest for Windows
        pin_memory=config.training.pin_memory,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.training.pin_memory,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.training.pin_memory,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    config = ProjectConfig()
    dataset = IUXRayDataset(split="train", config=config)
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample report: {sample['report']}")
        print(f"Sample labels: {sample['labels']}")
        print(f"Sample image shape: {sample['image'].shape}")
