"""
Preprocessing module for chest X-ray images and radiology report text.

This module provides data augmentation pipelines for images, converts raw chest X-rays
into the format expected by torchxrayvision models (single-channel, [-1024, 1024] range),
and cleans/processes raw report text for verification.
"""

import re
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Dict, List, Any, Union, Optional
from src.config import PATHOLOGY_LABELS

# ============================================================================
# Image Preprocessing & Augmentation
# ============================================================================

def get_train_transforms(input_size: int = 224) -> T.Compose:
    """
    Get image augmentation pipeline for training.

    Args:
        input_size: Target image size (width and height).

    Returns:
        torchvision Compose pipeline.
    """
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        # Normalize to torchxrayvision expected scale: [-1024, 1024]
        # Torchvision ToTensor outputs [0, 1]. Scaling it: (val * 2048) - 1024
        T.Lambda(lambda x: (x * 2048.0) - 1024.0),
    ])


def get_eval_transforms(input_size: int = 224) -> T.Compose:
    """
    Get standard image preprocessing pipeline for evaluation (validation/test/inference).

    Args:
        input_size: Target image size.

    Returns:
        torchvision Compose pipeline.
    """
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        # Normalize to torchxrayvision expected scale: [-1024, 1024]
        T.Lambda(lambda x: (x * 2048.0) - 1024.0),
    ])


def normalize_xray(image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
    """
    Convert a raw image (PIL or numpy) into a normalized tensor in [-1024, 1024] range.

    Args:
        image: Input image.

    Returns:
        Single-channel float tensor of shape (1, H, W).
    """
    if isinstance(image, Image.Image):
        # Convert to grayscale
        image = image.convert("L")
        image_np = np.array(image, dtype=np.float32)
    else:
        # If numpy array
        if len(image.shape) == 3:
            # RGB to Grayscale (Luma formula)
            image_np = (0.299 * image[:, :, 0] + 
                        0.587 * image[:, :, 1] + 
                        0.114 * image[:, :, 2]).astype(np.float32)
        else:
            image_np = image.astype(np.float32)

    # Normalize to [0, 1] range
    img_min, img_max = image_np.min(), image_np.max()
    if img_max > img_min:
        image_np = (image_np - img_min) / (img_max - img_min)
    else:
        image_np = np.zeros_like(image_np)

    # Rescale to [-1024, 1024] expected by torchxrayvision
    image_np = (image_np * 2048.0) - 1024.0
    
    # Convert to torch tensor (1, H, W)
    tensor = torch.from_numpy(image_np).unsqueeze(0).float()
    return tensor


# ============================================================================
# Text Preprocessing & Cleaning
# ============================================================================

def clean_report_text(text: str) -> str:
    """
    Clean and normalize radiology report text.

    Args:
        text: Raw report text.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
        
    # Lowercase
    cleaned = text.lower()
    
    # Remove metadata lines (e.g., "xml report", "study date:")
    cleaned = re.sub(r"(?:xml report|study date|patient id|accession number|physician):.*?\n", "", cleaned)
    
    # Normalize whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def extract_findings_section(report_text: str) -> str:
    """
    Extract the FINDINGS section from a report.

    Args:
        report_text: Raw or partially cleaned report.

    Returns:
        Findings section text.
    """
    # Regex search for findings section
    match = re.search(
        r"findings:(.*?)(?:impression:|comparison:|indication:|$)", 
        report_text, 
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return ""


def extract_impression_section(report_text: str) -> str:
    """
    Extract the IMPRESSION section from a report.

    Args:
        report_text: Raw or partially cleaned report.

    Returns:
        Impression section text.
    """
    # Regex search for impression section
    match = re.search(
        r"impression:(.*?)(?:findings:|comparison:|indication:|$)", 
        report_text, 
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
        
    # If not found directly, try matching from IMPRESSION: to the end
    match = re.search(r"impression:(.*?)$", report_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
        
    return ""


def binarize_labels(labels: torch.Tensor, uncertain_policy: str = "zero") -> torch.Tensor:
    """
    Convert label vectors containing uncertainty markers (-1) into binary vectors.

    Args:
        labels: Tensor with elements 0 (negative), 1 (positive), -1 (uncertain).
        uncertain_policy: Policy to resolve uncertainty.
            'zero': Map -1 to 0 (default, conservative clinical choice).
            'one': Map -1 to 1 (aggressive clinical choice).
            'ignore': Leave as is.

    Returns:
        Binarized torch.Tensor.
    """
    bin_labels = labels.clone()
    if uncertain_policy == "zero":
        bin_labels[bin_labels == -1.0] = 0.0
    elif uncertain_policy == "one":
        bin_labels[bin_labels == -1.0] = 1.0
    return bin_labels


if __name__ == "__main__":
    # Test cleaning
    sample_text = "FINDINGS: Cardiomegaly is present. IMPRESSION: Enlarged heart."
    print(f"Cleaned: {clean_report_text(sample_text)}")
    print(f"Findings: {extract_findings_section(sample_text)}")
    print(f"Impression: {extract_impression_section(sample_text)}")
