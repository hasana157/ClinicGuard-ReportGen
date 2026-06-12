"""
Dataset loaders for MIMIC-CXR, PadChest, and IU X-Ray.

MIMIC-CXR and PadChest are protected datasets and must be downloaded by the
user after approval. The loader validates local dataset structure and fails
with actionable messages when those datasets are not configured. IU X-Ray is
kept as an explicit free fallback for demos and offline development.
"""

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import PATHOLOGY_LABELS, ProjectConfig


SUPPORTED_DATASETS = ("MIMIC-CXR", "PADCHEST", "IU-XRAY")

DATASET_ALIASES = {
    "MIMIC": "MIMIC-CXR",
    "MIMIC-CXR": "MIMIC-CXR",
    "MIMIC-CXR-JPG": "MIMIC-CXR",
    "PAD-CHEST": "PADCHEST",
    "PADCHEST": "PADCHEST",
    "IU": "IU-XRAY",
    "IUXRAY": "IU-XRAY",
    "IU-XRAY": "IU-XRAY",
    "OPEN-I": "IU-XRAY",
}

# Synonym dictionary for pathology keyword extraction.
PATHOLOGY_SYNONYMS = {
    "Atelectasis": [
        "atelectasis",
        "atelectases",
        "subsegmental atelectasis",
        "platelike atelectasis",
        "discoid atelectasis",
    ],
    "Cardiomegaly": [
        "cardiomegaly",
        "enlarged heart",
        "cardiac enlargement",
        "enlarged cardiac silhouette",
        "prominent cardiac silhouette",
    ],
    "Consolidation": [
        "consolidation",
        "consolidations",
        "airspace consolidation",
        "focal consolidation",
    ],
    "Edema": ["edema", "pulmonary edema", "vascular congestion", "fluid overload"],
    "Effusion": [
        "effusion",
        "effusions",
        "pleural effusion",
        "pleural effusions",
        "fluid in the pleural space",
    ],
    "Emphysema": ["emphysema", "emphysematous changes", "hyperinflation"],
    "Fibrosis": ["fibrosis", "scarring", "fibrotic", "interstitial scarring"],
    "Hernia": ["hernia", "hiatal hernia"],
    "Infiltration": [
        "infiltration",
        "infiltrations",
        "infiltrates",
        "infiltrate",
        "interstitial infiltrate",
        "airspace disease",
        "lung opacity",
    ],
    "Mass": ["mass", "masses", "large nodule", "tumor"],
    "Nodule": ["nodule", "nodules", "nodular opacity", "nodular opacities"],
    "Pleural_Thickening": [
        "pleural thickening",
        "pleural thickening/scarring",
        "apical thickening",
        "pleural other",
    ],
    "Pneumonia": [
        "pneumonia",
        "pneumonias",
        "infectious process",
        "consolidation suggesting pneumonia",
    ],
    "Pneumothorax": ["pneumothorax", "pneumothoraces", "collapsed lung"],
}

LABEL_COLUMN_ALIASES = {
    "Atelectasis": ["Atelectasis", "atelectasis"],
    "Cardiomegaly": ["Cardiomegaly", "cardiomegaly"],
    "Consolidation": ["Consolidation", "consolidation"],
    "Edema": ["Edema", "Pulmonary Edema", "edema"],
    "Effusion": ["Effusion", "Pleural Effusion", "effusion"],
    "Emphysema": ["Emphysema", "emphysema"],
    "Fibrosis": ["Fibrosis", "fibrosis"],
    "Hernia": ["Hernia", "Hiatal Hernia", "hernia"],
    "Infiltration": ["Infiltration", "Lung Opacity", "infiltration"],
    "Mass": ["Mass", "mass"],
    "Nodule": ["Nodule", "nodule"],
    "Pleural_Thickening": [
        "Pleural Thickening",
        "Pleural_Thickening",
        "Pleural Other",
        "pleural_thickening",
    ],
    "Pneumonia": ["Pneumonia", "pneumonia"],
    "Pneumothorax": ["Pneumothorax", "pneumothorax"],
}


class DatasetNotConfiguredError(FileNotFoundError):
    """Raised when a protected dataset path is missing or incomplete."""


def normalize_dataset_name(dataset_name: Optional[str]) -> str:
    """Normalize CLI/config dataset aliases to canonical names."""
    raw = dataset_name or "MIMIC-CXR"
    key = re.sub(r"[\s_]+", "-", raw.strip().upper())
    normalized = DATASET_ALIASES.get(key)
    if normalized is None:
        allowed = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose one of: {allowed}.")
    return normalized


class MedicalReportDataset(Dataset):
    """Universal chest X-ray report dataset for MIMIC-CXR, PadChest, and IU X-Ray."""

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        split: str = "train",
        transform: Optional[Any] = None,
        config: Optional[ProjectConfig] = None,
        sample_limit: Optional[int] = None,
    ):
        self.config = config or ProjectConfig()
        self.dataset_name = normalize_dataset_name(
            dataset_name or self.config.data.primary_dataset
        )
        self.split = _normalize_split(split)
        self.transform = transform
        self.sample_limit = sample_limit
        self.samples: List[Dict[str, Any]] = []

        print(f"Loading {self.dataset_name} dataset ({self.split} split)...")

        if self.dataset_name == "MIMIC-CXR":
            self._load_mimic_cxr()
        elif self.dataset_name == "PADCHEST":
            self._load_padchest()
        elif self.dataset_name == "IU-XRAY":
            self._load_iu_xray()

        limit = (
            self.sample_limit
            if self.sample_limit is not None
            else self.config.data.max_samples_per_split
        )
        if limit is not None:
            self.samples = self.samples[: max(0, int(limit))]

        if not self.samples:
            raise RuntimeError(
                f"No samples loaded for {self.dataset_name} ({self.split}). "
                "Check dataset paths, metadata files, and split configuration."
            )

        print(f"Loaded {len(self.samples)} samples from {self.dataset_name}.")

    def _load_mimic_cxr(self) -> None:
        """Load MIMIC-CXR-JPG images and reports from a local approved copy."""
        root = Path(self.config.data.mimic_cxr_path)
        if not root.exists():
            raise DatasetNotConfiguredError(
                f"MIMIC-CXR not found at {root}. Download MIMIC-CXR-JPG after "
                "PhysioNet approval, or pass --dataset IU-XRAY for the free fallback."
            )

        files_dir = root / "files"
        if not files_dir.exists():
            raise DatasetNotConfiguredError(
                f"MIMIC-CXR structure incomplete at {root}: expected a files/ directory."
            )

        report_lookup = _load_mimic_report_lookup(root)
        label_lookup = _load_mimic_label_lookup(root)
        metadata_csv = _first_existing_file(
            root,
            [
                "mimic-cxr-2.0.0-metadata.csv",
                "mimic-cxr-metadata.csv",
                "metadata.csv",
            ],
            ["*metadata*.csv"],
        )

        if metadata_csv is None:
            self._load_mimic_from_image_tree(root, report_lookup, label_lookup)
            return

        df = pd.read_csv(metadata_csv, low_memory=False)
        df = _merge_mimic_split(root, df)
        df = _filter_dataframe_split(df, self.split, self.config)

        for _, row in df.iterrows():
            subject_id = _id_string(_row_value(row, ["subject_id", "SubjectID"]))
            study_id = _id_string(_row_value(row, ["study_id", "StudyID"]))
            if not subject_id or not study_id:
                continue

            image_path = _mimic_image_path(root, row, subject_id, study_id)
            if image_path is None:
                continue

            labels = _lookup_label_vector(label_lookup, subject_id, study_id)
            report = _lookup_mimic_report(report_lookup, root, subject_id, study_id)
            if not report and labels is not None:
                report = _report_from_labels(labels, "MIMIC-CXR")
            if not report:
                report = "FINDINGS:\nReport text unavailable.\n\nIMPRESSION:\nNot provided."

            self.samples.append(
                _make_sample(
                    image_path=str(image_path),
                    report=report,
                    labels=labels,
                    sample_id=_mimic_sample_id(row, subject_id, study_id),
                    dataset="MIMIC-CXR",
                )
            )

    def _load_mimic_from_image_tree(
        self,
        root: Path,
        report_lookup: Dict[Any, str],
        label_lookup: Dict[Any, List[float]],
    ) -> None:
        """Fallback for MIMIC copies without a metadata CSV."""
        image_paths = sorted((root / "files").glob("p*/p*/s*/*.jpg"))
        image_paths = _split_sequence(image_paths, self.split, self.config)

        for image_path in image_paths:
            study_dir = image_path.parent
            subject_dir = study_dir.parent
            subject_id = subject_dir.name.lstrip("p")
            study_id = study_dir.name.lstrip("s")
            labels = _lookup_label_vector(label_lookup, subject_id, study_id)
            report = _lookup_mimic_report(report_lookup, root, subject_id, study_id)
            if not report and labels is not None:
                report = _report_from_labels(labels, "MIMIC-CXR")

            self.samples.append(
                _make_sample(
                    image_path=str(image_path),
                    report=report,
                    labels=labels,
                    sample_id=f"MIMIC-{study_id}-{image_path.stem}",
                    dataset="MIMIC-CXR",
                )
            )

    def _load_padchest(self) -> None:
        """Load PadChest images and metadata from a local approved copy."""
        root = Path(self.config.data.padchest_path)
        if not root.exists():
            raise DatasetNotConfiguredError(
                f"PadChest not found at {root}. Request BIMCV access and extract "
                "the dataset locally, or pass --dataset IU-XRAY for the free fallback."
            )

        metadata_csv = _first_existing_file(
            root,
            [
                "PADCHEST_chest_x_ray_images_labels_01_v2.csv",
                "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
                "padchest.csv",
            ],
            ["PADCHEST*labels*.csv", "*padchest*.csv", "*.csv"],
        )
        if metadata_csv is None:
            raise DatasetNotConfiguredError(
                f"PadChest metadata CSV not found under {root}."
            )

        df = pd.read_csv(metadata_csv, low_memory=False)
        df = _filter_dataframe_split(df, self.split, self.config)

        for _, row in df.iterrows():
            image_id = _row_value(
                row,
                [
                    "ImageID",
                    "ImageId",
                    "image_id",
                    "filename",
                    "Filename",
                    "file_name",
                    "image",
                ],
            )
            if _is_missing(image_id):
                continue

            image_path = _padchest_image_path(root, str(image_id))
            if image_path is None:
                continue

            report = _padchest_report_text(row)
            labels = _labels_from_row(row)
            if labels is None:
                labels = extract_labels_from_report(report, PATHOLOGY_LABELS)
            if not report:
                report = _report_from_labels(labels, "PadChest")

            self.samples.append(
                _make_sample(
                    image_path=str(image_path),
                    report=report,
                    labels=labels,
                    sample_id=f"PADCHEST-{Path(str(image_id)).stem}",
                    dataset="PADCHEST",
                )
            )

    def _load_iu_xray(self) -> None:
        """Load IU X-Ray from HuggingFace, with an offline mock fallback."""
        try:
            dataset = _load_first_hf_dataset(
                [
                    self.config.data.iu_xray_hf_dataset,
                    self.config.data.iu_xray_alt_dataset,
                ]
            )
            split_indices = _split_indices(len(dataset), self.split, self.config)

            for idx in split_indices:
                item = dataset[int(idx)]
                image = _first_item_value(
                    item,
                    ["image", "image_1", "frontal_image", "frontal", "images"],
                )
                if isinstance(image, (list, tuple)):
                    image = image[0] if image else None

                findings = _first_item_value(
                    item, ["findings", "Findings", "report", "caption", "text"]
                )
                impression = _first_item_value(item, ["impression", "Impression"])
                report_text = _compose_report_text(findings, impression)
                labels = extract_labels_from_report(report_text, PATHOLOGY_LABELS)

                self.samples.append(
                    _make_sample(
                        image=image,
                        report=report_text,
                        labels=labels,
                        sample_id=f"IU-{int(idx)}",
                        dataset="IU-XRAY",
                    )
                )
        except Exception as exc:
            if not self.config.data.allow_iu_mock_fallback:
                raise
            print(
                "IU X-Ray load failed or the environment is offline: "
                f"{exc}. Generating mock dataset for execution."
            )
            self._generate_mock_dataset()

    def _generate_mock_dataset(self) -> None:
        """Generate a small deterministic mock dataset for offline smoke tests."""
        split_offset = {"train": 0, "val": 1, "test": 2}[self.split]
        rng = np.random.default_rng(self.config.training.seed + split_offset)
        num_samples = 40 if self.split == "train" else 10

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
            image = Image.fromarray(rng.integers(0, 255, (224, 224), dtype=np.uint8))
            report_idx = int(rng.integers(0, len(mock_findings)))
            report_text = _compose_report_text(
                mock_findings[report_idx], mock_impressions[report_idx]
            )
            self.samples.append(
                _make_sample(
                    image=image,
                    report=report_text,
                    labels=extract_labels_from_report(report_text, PATHOLOGY_LABELS),
                    sample_id=f"MOCK-{self.split}-{i}",
                    dataset="IU-XRAY",
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = _load_sample_image(sample).convert("L")

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image = image.resize(
                (self.config.vision.input_size, self.config.vision.input_size)
            )
            image_np = np.array(image, dtype=np.float32) / 255.0
            image_np = (image_np * 2048.0) - 1024.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()

        if isinstance(image_tensor, Image.Image):
            image_np = np.array(image_tensor.convert("L"), dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy((image_np * 2048.0) - 1024.0).unsqueeze(0)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)

        labels = sample.get("labels")
        if labels is None:
            labels = extract_labels_from_report(sample["report"], PATHOLOGY_LABELS)

        return {
            "image": image_tensor.float(),
            "report": sample["report"],
            "findings": sample["findings"],
            "impression": sample["impression"],
            "labels": torch.tensor(labels, dtype=torch.float32),
            "dataset": sample["dataset"],
            "metadata": {
                "id": sample["id"],
                "dataset": sample["dataset"],
                "image_path": sample.get("image_path"),
            },
        }


class IUXRayDataset(MedicalReportDataset):
    """Backward-compatible IU X-Ray dataset wrapper."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Any] = None,
        config: Optional[ProjectConfig] = None,
        sample_limit: Optional[int] = None,
    ):
        super().__init__(
            dataset_name="IU-XRAY",
            split=split,
            transform=transform,
            config=config,
            sample_limit=sample_limit,
        )


class MIMICCXRLoader(MedicalReportDataset):
    """Backward-compatible MIMIC-CXR dataset wrapper."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Any] = None,
        config: Optional[ProjectConfig] = None,
        sample_limit: Optional[int] = None,
    ):
        super().__init__(
            dataset_name="MIMIC-CXR",
            split=split,
            transform=transform,
            config=config,
            sample_limit=sample_limit,
        )


class PadChestLoader(MedicalReportDataset):
    """Backward-compatible PadChest dataset wrapper."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Any] = None,
        config: Optional[ProjectConfig] = None,
        sample_limit: Optional[int] = None,
    ):
        super().__init__(
            dataset_name="PADCHEST",
            split=split,
            transform=transform,
            config=config,
            sample_limit=sample_limit,
        )


def extract_labels_from_report(report_text: str, pathology_labels: List[str]) -> List[float]:
    """
    Extract labels from report text using keyword matching and simple negation detection.

    Returns 1.0 for positive, 0.0 for negative/not mentioned, and -1.0 for uncertain.
    """
    report_lower = (report_text or "").lower()
    labels = [0.0] * len(pathology_labels)

    negation_patterns = [
        r"\bno\s+evidence\s+of\s+{finding}",
        r"\bno\s+{finding}",
        r"\bwithout\s+{finding}",
        r"\bfree\s+of\s+{finding}",
        r"\bnegative\s+for\s+{finding}",
        r"\bclear\s+of\s+{finding}",
        r"\bnot\s+seen\b",
        r"\babsent\b",
        r"\bruled\s+out\b",
    ]
    uncertain_patterns = [
        r"\brule\s+out\s+{finding}",
        r"\bpossible\s+{finding}",
        r"\bpossibly\s+{finding}",
        r"\bsuggestive\s+of\s+{finding}",
        r"\bmay\s+represent\s+{finding}",
        r"\bcannot\s+exclude\s+{finding}",
        r"\bborderline\s+{finding}",
        r"\bcompatible\s+with\s+{finding}",
        r"\bquery\s+{finding}",
    ]

    for idx, pathology in enumerate(pathology_labels):
        synonyms = PATHOLOGY_SYNONYMS.get(pathology, [pathology.lower()])
        present = False
        uncertain = False

        for synonym in synonyms:
            if synonym not in report_lower:
                continue

            finding_pattern = re.escape(synonym)
            negated = any(
                re.search(pattern.format(finding=finding_pattern), report_lower)
                for pattern in negation_patterns
            )
            if negated:
                continue

            uncertain = any(
                re.search(pattern.format(finding=finding_pattern), report_lower)
                for pattern in uncertain_patterns
            )
            present = True
            break

        labels[idx] = -1.0 if present and uncertain else 1.0 if present else 0.0

    return labels


def get_report_sections(report_text: str) -> Dict[str, str]:
    """Split a radiology report into common report sections."""
    text = report_text or ""
    sections = {"INDICATION": "", "FINDINGS": "", "IMPRESSION": ""}

    findings_match = re.search(
        r"FINDINGS:(.*?)(?:IMPRESSION:|$)", text, re.DOTALL | re.IGNORECASE
    )
    impression_match = re.search(
        r"IMPRESSION:(.*?)$", text, re.DOTALL | re.IGNORECASE
    )

    if findings_match:
        sections["FINDINGS"] = findings_match.group(1).strip()
    if impression_match:
        sections["IMPRESSION"] = impression_match.group(1).strip()
    return sections


def get_dataloaders(
    config: ProjectConfig,
    dataset_name: Optional[str] = None,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    sample_limit: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoader instances for train, validation, and test splits."""
    selected_dataset = normalize_dataset_name(dataset_name or config.data.primary_dataset)

    train_set = MedicalReportDataset(
        dataset_name=selected_dataset,
        split="train",
        transform=train_transform,
        config=config,
        sample_limit=sample_limit,
    )
    val_set = MedicalReportDataset(
        dataset_name=selected_dataset,
        split="val",
        transform=eval_transform,
        config=config,
        sample_limit=sample_limit,
    )
    test_set = MedicalReportDataset(
        dataset_name=selected_dataset,
        split="test",
        transform=eval_transform,
        config=config,
        sample_limit=sample_limit,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
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


def _normalize_split(split: str) -> str:
    value = (split or "train").strip().lower()
    aliases = {"validation": "val", "validate": "val", "dev": "val"}
    value = aliases.get(value, value)
    if value not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    return value


def _split_indices(n_total: int, split: str, config: ProjectConfig) -> np.ndarray:
    indices = np.arange(n_total)
    rng = np.random.default_rng(config.training.seed)
    rng.shuffle(indices)

    n_train = int(n_total * config.data.train_ratio)
    n_val = int(n_total * config.data.val_ratio)
    if split == "train":
        return indices[:n_train]
    if split == "val":
        return indices[n_train : n_train + n_val]
    return indices[n_train + n_val :]


def _split_sequence(items: List[Any], split: str, config: ProjectConfig) -> List[Any]:
    return [items[int(i)] for i in _split_indices(len(items), split, config)]


def _filter_dataframe_split(
    df: pd.DataFrame, split: str, config: ProjectConfig
) -> pd.DataFrame:
    split_col = _first_column(df, ["split", "Split", "dataset_split"])
    if split_col is not None:
        values = df[split_col].astype(str).str.lower().str.strip()
        if split == "train":
            mask = values.isin(["train", "training"])
        elif split == "val":
            mask = values.isin(["val", "valid", "validate", "validation"])
        else:
            mask = values.isin(["test", "testing"])
        filtered = df[mask]
        if not filtered.empty:
            return filtered.reset_index(drop=True)

    split_idx = _split_indices(len(df), split, config)
    return df.iloc[split_idx].reset_index(drop=True)


def _first_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        found = lower_map.get(candidate.lower())
        if found:
            return found
    return None


def _first_existing_file(
    root: Path, filenames: Iterable[str], patterns: Iterable[str]
) -> Optional[Path]:
    for filename in filenames:
        candidate = root / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    for pattern in patterns:
        matches = sorted(p for p in root.glob(pattern) if p.is_file())
        if matches:
            return matches[0]
    return None


def _merge_mimic_split(root: Path, df: pd.DataFrame) -> pd.DataFrame:
    if _first_column(df, ["split", "Split"]) is not None:
        return df

    split_csv = _first_existing_file(
        root,
        ["mimic-cxr-2.0.0-split.csv", "mimic-cxr-split.csv", "split.csv"],
        ["*split*.csv"],
    )
    if split_csv is None:
        return df

    split_df = pd.read_csv(split_csv, low_memory=False)
    join_cols = [col for col in ["dicom_id", "study_id", "subject_id"] if col in df.columns and col in split_df.columns]
    if not join_cols:
        return df
    return df.merge(split_df[join_cols + ["split"]].drop_duplicates(), on=join_cols, how="left")


def _load_mimic_report_lookup(root: Path) -> Dict[Any, str]:
    report_csv = _first_existing_file(
        root,
        ["mimic-cxr-reports.csv", "mimic-cxr-2.0.0-reports.csv", "reports.csv"],
        ["*reports*.csv", "*report*.csv"],
    )
    if report_csv is None:
        return {}

    df = pd.read_csv(report_csv, low_memory=False)
    lookup: Dict[Any, str] = {}
    for _, row in df.iterrows():
        subject_id = _id_string(_row_value(row, ["subject_id", "SubjectID"]))
        study_id = _id_string(_row_value(row, ["study_id", "StudyID"]))
        report = _report_from_row(row)
        if not report or not study_id:
            continue
        lookup[study_id] = report
        if subject_id:
            lookup[(subject_id, study_id)] = report
    return lookup


def _load_mimic_label_lookup(root: Path) -> Dict[Any, List[float]]:
    labels_csv = _first_existing_file(
        root,
        [
            "mimic-cxr-2.0.0-chexpert.csv",
            "mimic-cxr-chexpert.csv",
            "chexpert.csv",
            "mimic-cxr-2.0.0-negbio.csv",
        ],
        ["*chexpert*.csv", "*negbio*.csv"],
    )
    if labels_csv is None:
        return {}

    df = pd.read_csv(labels_csv, low_memory=False)
    lookup: Dict[Any, List[float]] = {}
    for _, row in df.iterrows():
        subject_id = _id_string(_row_value(row, ["subject_id", "SubjectID"]))
        study_id = _id_string(_row_value(row, ["study_id", "StudyID"]))
        labels = _labels_from_row(row)
        if labels is None or not study_id:
            continue
        lookup[study_id] = labels
        if subject_id:
            lookup[(subject_id, study_id)] = labels
    return lookup


def _lookup_mimic_report(
    report_lookup: Dict[Any, str], root: Path, subject_id: str, study_id: str
) -> str:
    report = report_lookup.get((subject_id, study_id)) or report_lookup.get(study_id)
    if report:
        return report

    prefix = subject_id[:2]
    candidates = [
        root / "files" / f"p{prefix}" / f"p{subject_id}" / f"s{study_id}.txt",
        root / "reports" / f"p{prefix}" / f"p{subject_id}" / f"s{study_id}.txt",
        root / "files" / f"p{prefix}" / f"p{subject_id}" / f"s{study_id}" / f"s{study_id}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def _lookup_label_vector(
    label_lookup: Dict[Any, List[float]], subject_id: str, study_id: str
) -> Optional[List[float]]:
    return label_lookup.get((subject_id, study_id)) or label_lookup.get(study_id)


def _mimic_image_path(
    root: Path, row: pd.Series, subject_id: str, study_id: str
) -> Optional[Path]:
    path_value = _row_value(row, ["image_path", "path", "jpg_path", "filepath"])
    resolved = _resolve_existing_path(root, path_value)
    if resolved is not None:
        return resolved

    dicom_id = _id_string(_row_value(row, ["dicom_id", "DICOM_ID", "image_id"]))
    study_dir = root / "files" / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}"
    if dicom_id:
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = study_dir / f"{dicom_id}{ext}"
            if candidate.exists():
                return candidate

    if study_dir.exists():
        matches = sorted(
            list(study_dir.glob("*.jpg"))
            + list(study_dir.glob("*.jpeg"))
            + list(study_dir.glob("*.png"))
        )
        if matches:
            return matches[0]
    return None


def _mimic_sample_id(row: pd.Series, subject_id: str, study_id: str) -> str:
    dicom_id = _id_string(_row_value(row, ["dicom_id", "DICOM_ID", "image_id"]))
    if dicom_id:
        return f"MIMIC-{study_id}-{dicom_id}"
    return f"MIMIC-{subject_id}-{study_id}"


def _padchest_image_path(root: Path, image_id: str) -> Optional[Path]:
    image_id = image_id.strip()
    resolved = _resolve_existing_path(root, image_id)
    if resolved is not None:
        return resolved

    image_dirs = [
        root / "image_dir",
        root / "images",
        root / "Images",
        root / "png",
        root / "jpg",
        root,
    ]
    names = [image_id]
    if not Path(image_id).suffix:
        names.extend([f"{image_id}.png", f"{image_id}.jpg", f"{image_id}.jpeg"])

    for image_dir in image_dirs:
        for name in names:
            candidate = image_dir / name
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _padchest_report_text(row: pd.Series) -> str:
    report = _row_value(
        row,
        [
            "Report",
            "report",
            "ReportText",
            "findings",
            "Findings",
            "Report_ENG",
            "Labels",
            "labels",
        ],
    )
    return "" if _is_missing(report) else str(report)


def _labels_from_row(row: pd.Series) -> Optional[List[float]]:
    labels = [0.0] * len(PATHOLOGY_LABELS)
    found_any = False

    for idx, label in enumerate(PATHOLOGY_LABELS):
        for column in LABEL_COLUMN_ALIASES.get(label, [label]):
            if column not in row.index:
                continue
            value = row[column]
            if _is_missing(value):
                continue
            found_any = True
            labels[idx] = _coerce_label_value(value)
            break

    label_text = _row_value(row, ["Labels", "labels", "label", "diagnosis"])
    if not _is_missing(label_text):
        found_any = True
        text_labels = extract_labels_from_report(_labels_text_to_report(label_text), PATHOLOGY_LABELS)
        labels = [
            _merge_label_values(existing, text_label)
            for existing, text_label in zip(labels, text_labels)
        ]

    return labels if found_any else None


def _coerce_label_value(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip().lower()
        if text in {"positive", "present", "true", "yes"}:
            return 1.0
        if text in {"uncertain", "maybe", "possible"}:
            return -1.0
        return 0.0

    if numeric > 0:
        return 1.0
    if numeric < 0:
        return -1.0
    return 0.0


def _merge_label_values(existing: float, incoming: float) -> float:
    if existing == 1.0 or incoming == 1.0:
        return 1.0
    if existing == -1.0 or incoming == -1.0:
        return -1.0
    return 0.0


def _labels_text_to_report(value: Any) -> str:
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            text = " ".join(str(item) for item in parsed)
    except (ValueError, SyntaxError):
        pass
    return text.replace("_", " ")


def _report_from_row(row: pd.Series) -> str:
    report = _row_value(row, ["text", "Text", "report", "Report", "report_text"])
    if not _is_missing(report):
        return str(report)

    findings = _row_value(row, ["findings", "Findings", "FINDINGS"])
    impression = _row_value(row, ["impression", "Impression", "IMPRESSION"])
    if _is_missing(findings) and _is_missing(impression):
        return ""
    return _compose_report_text(findings, impression)


def _compose_report_text(findings: Any, impression: Any) -> str:
    findings_text = "" if _is_missing(findings) else str(findings).strip()
    impression_text = "" if _is_missing(impression) else str(impression).strip()
    return f"FINDINGS:\n{findings_text}\n\nIMPRESSION:\n{impression_text}".strip()


def _report_from_labels(labels: Optional[List[float]], dataset_name: str) -> str:
    labels = labels or [0.0] * len(PATHOLOGY_LABELS)
    positives = [
        label.replace("_", " ")
        for label, value in zip(PATHOLOGY_LABELS, labels)
        if value == 1.0
    ]
    uncertain = [
        label.replace("_", " ")
        for label, value in zip(PATHOLOGY_LABELS, labels)
        if value == -1.0
    ]

    if positives:
        findings = f"{dataset_name} metadata marks the following findings as present: {', '.join(positives)}."
    else:
        findings = f"{dataset_name} metadata provides no positive finding labels."
    if uncertain:
        findings += f" Uncertain labels: {', '.join(uncertain)}."
    return f"FINDINGS:\n{findings}\n\nIMPRESSION:\nSee structured labels."


def _make_sample(
    *,
    report: str,
    labels: Optional[List[float]],
    sample_id: str,
    dataset: str,
    image_path: Optional[str] = None,
    image: Optional[Any] = None,
) -> Dict[str, Any]:
    sections = get_report_sections(report)
    if labels is None:
        labels = extract_labels_from_report(report, PATHOLOGY_LABELS)
    return {
        "image_path": image_path,
        "image": image,
        "report": report,
        "findings": sections.get("FINDINGS", ""),
        "impression": sections.get("IMPRESSION", ""),
        "labels": labels,
        "id": sample_id,
        "dataset": dataset,
    }


def _load_sample_image(sample: Dict[str, Any]) -> Image.Image:
    if sample.get("image_path"):
        return Image.open(sample["image_path"])

    image = sample.get("image")
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
        arr = np.squeeze(arr)
        return Image.fromarray(arr.astype(np.uint8))

    raise ValueError(f"Sample {sample.get('id')} has no readable image.")


def _load_first_hf_dataset(dataset_ids: Iterable[str]) -> Any:
    from datasets import load_dataset

    last_error: Optional[Exception] = None
    for dataset_id in dict.fromkeys(dataset_ids):
        try:
            print(f"Loading {dataset_id} from HuggingFace...")
            return load_dataset(dataset_id, split="train")
        except Exception as exc:
            last_error = exc
            print(f"Could not load {dataset_id}: {exc}")
    if last_error is not None:
        raise last_error
    raise RuntimeError("No HuggingFace dataset IDs configured for IU X-Ray.")


def _row_value(row: pd.Series, candidates: Iterable[str]) -> Any:
    lower_map = {str(col).lower(): col for col in row.index}
    for candidate in candidates:
        column = candidate if candidate in row.index else lower_map.get(candidate.lower())
        if column is None:
            continue
        value = row[column]
        if not _is_missing(value):
            return value
    return None


def _first_item_value(item: Dict[str, Any], candidates: Iterable[str]) -> Any:
    lower_map = {str(key).lower(): key for key in item.keys()}
    for candidate in candidates:
        key = candidate if candidate in item else lower_map.get(candidate.lower())
        if key is None:
            continue
        value = item[key]
        if not _is_missing(value):
            return value
    return None


def _resolve_existing_path(root: Path, value: Any) -> Optional[Path]:
    if _is_missing(value):
        return None
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else [root / path]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _id_string(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) > 1 and text[0] in {"p", "s"} and text[1:].isdigit():
        return text[1:]
    return text


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"nan", "none", "null"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    config = ProjectConfig()
    dataset = MedicalReportDataset(dataset_name="IU-XRAY", split="train", config=config, sample_limit=2)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample report: {sample['report']}")
    print(f"Sample labels: {sample['labels']}")
    print(f"Sample image shape: {sample['image'].shape}")
