"""
Central configuration for the Medical Report Generation system.

All hyperparameters, model names, paths, and thresholds are defined here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# Path Configuration
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
SAMPLE_CASES_DIR = os.path.join(DATA_DIR, "sample_cases")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation")
SAMPLE_OUTPUTS_DIR = os.path.join(EVALUATION_DIR, "sample_outputs")


# ============================================================================
# Dataset Configuration
# ============================================================================

# IU X-Ray (Indiana University) — Primary dataset
IU_XRAY_HF_DATASET = "ykumards/open-i"  # HuggingFace dataset identifier
IU_XRAY_ALT_DATASET = "dz-osamu/IU-Xray"  # Alternative HuggingFace source

# Dataset splits
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class VisionEncoderConfig:
    """Configuration for the vision encoder backbone."""
    model_name: str = "densenet121-res224-mimic_ch"
    input_size: int = 224
    feature_dim: int = 1024
    spatial_size: int = 7  # Spatial feature map: 7x7
    num_pathologies: int = 18  # torchxrayvision DenseNet outputs 18 pathologies
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.3


@dataclass
class ClassificationConfig:
    """Configuration for the pathology classification head."""
    num_classes: int = 14  # Primary pathology labels
    confidence_threshold_positive: float = 0.75  # Assert finding
    confidence_threshold_uncertain: float = 0.50  # Hedge language
    confidence_threshold_refusal: float = 0.50  # Below = abstain
    

@dataclass
class ReportGeneratorConfig:
    """Configuration for constrained report generation."""
    evidence_threshold: float = 0.75  # Minimum confidence for positive assertion
    uncertainty_threshold: float = 0.50  # Below this = refuse
    max_findings_per_report: int = 10
    include_patient_history: bool = True
    include_prior_reports: bool = True
    refusal_token: str = "[INSUFFICIENT_EVIDENCE]"


@dataclass
class GradCAMConfig:
    """Configuration for Grad-CAM grounding."""
    target_layer: str = "features.denseblock4"  # Last dense block
    colormap: str = "jet"
    alpha_overlay: float = 0.4
    bbox_threshold: float = 0.5  # Heatmap threshold for bounding box extraction
    output_size: tuple = (224, 224)


@dataclass
class HallucinationDetectorConfig:
    """Configuration for hallucination detection."""
    semantic_similarity_threshold: float = 0.70
    claim_extraction_method: str = "sentence"  # "sentence" or "clause"
    cross_reference_sources: List[str] = field(
        default_factory=lambda: ["visual", "history", "prior"]
    )


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 30
    early_stopping_patience: int = 5
    scheduler: str = "cosine"  # "cosine" or "step"
    warmup_epochs: int = 2
    gradient_clip_norm: float = 1.0
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True  # Use AMP for faster training
    checkpoint_dir: str = MODELS_DIR
    log_interval: int = 10  # Log every N batches


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    hallucination_penalty_weight: float = 5.0  # Hallucinations penalized 5x
    compute_bleu: bool = True
    compute_rouge: bool = True
    compute_meteor: bool = True
    num_test_samples: Optional[int] = None  # None = use all test samples


# ============================================================================
# Pathology Labels
# ============================================================================

# 14 primary pathology classes for chest X-ray
PATHOLOGY_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

# Extended labels from torchxrayvision (18 classes)
TORCHXRAYVISION_LABELS = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]

# Mapping from torchxrayvision index to our primary labels
TXV_TO_PRIMARY = {i: label for i, label in enumerate(TORCHXRAYVISION_LABELS)
                  if label in PATHOLOGY_LABELS}


# ============================================================================
# Report Sections
# ============================================================================

REPORT_SECTIONS = ["INDICATION", "COMPARISON", "FINDINGS", "IMPRESSION"]


# ============================================================================
# Composite Config
# ============================================================================

@dataclass
class ProjectConfig:
    """Master configuration combining all sub-configs."""
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    report: ReportGeneratorConfig = field(default_factory=ReportGeneratorConfig)
    gradcam: GradCAMConfig = field(default_factory=GradCAMConfig)
    hallucination: HallucinationDetectorConfig = field(default_factory=HallucinationDetectorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def get_config() -> ProjectConfig:
    """Get the default project configuration."""
    return ProjectConfig()


def ensure_directories():
    """Create all necessary project directories."""
    dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR,
        SAMPLE_CASES_DIR, MODELS_DIR, RESULTS_DIR, REPORTS_DIR,
        EVALUATION_DIR, SAMPLE_OUTPUTS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
