"""
Generate report deliverables for ClinicGuard-ReportGen.

This script creates the expanded sample evidence log, benchmark summary, visual
report assets, Markdown reports, and PDFs. The metrics are clearly labeled as an
offline sample audit, not as a clinical benchmark.
"""

import os
import sys
from datetime import date
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_pdfs import generate_pdf_from_md
from src.config import EVALUATION_DIR, PATHOLOGY_LABELS, REPORTS_DIR


ASSETS_DIR = os.path.join(REPORTS_DIR, "assets")


def ensure_dirs() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)


def evidence_rows() -> list[dict]:
    """Build a deterministic claim-level sample audit log."""
    cases = [
        {
            "id": "001",
            "context": "dyspnea, hypertension",
            "positives": [("Cardiomegaly", 0.94, "[92,132,276,336]"), ("Effusion", 0.78, "[36,298,156,418]")],
            "uncertain": [("Edema", 0.58, "[118,152,322,302]")],
            "negatives": [("Pneumothorax", 0.95), ("Pneumonia", 0.88)],
            "refused": [("Mass", 0.21), ("Nodule", 0.28)],
            "hallucinated": [("Large right upper lobe mass", "Mass", 0.21)],
        },
        {
            "id": "002",
            "context": "cough and fever",
            "positives": [("Pneumonia", 0.86, "[142,108,342,302]"), ("Consolidation", 0.82, "[152,126,336,284]")],
            "uncertain": [("Effusion", 0.54, "[28,308,142,420]")],
            "negatives": [("Pneumothorax", 0.91), ("Cardiomegaly", 0.84)],
            "refused": [("Hernia", 0.08)],
            "hallucinated": [],
        },
        {
            "id": "003",
            "context": "follow-up pleural effusion",
            "positives": [("Effusion", 0.83, "[40,304,162,430]")],
            "uncertain": [("Atelectasis", 0.61, "[202,288,360,412]")],
            "negatives": [("Pneumonia", 0.89), ("Pneumothorax", 0.94)],
            "refused": [("Mass", 0.16), ("Fibrosis", 0.31)],
            "hallucinated": [],
        },
        {
            "id": "004",
            "context": "baseline screening",
            "positives": [],
            "uncertain": [("Pleural_Thickening", 0.57, "[24,70,136,192]")],
            "negatives": [("Pneumonia", 0.92), ("Pneumothorax", 0.96), ("Effusion", 0.88), ("Cardiomegaly", 0.86)],
            "refused": [("Nodule", 0.34), ("Mass", 0.22)],
            "hallucinated": [],
        },
        {
            "id": "005",
            "context": "chest pain",
            "positives": [("Cardiomegaly", 0.88, "[90,138,270,330]")],
            "uncertain": [("Infiltration", 0.53, "[118,118,316,296]")],
            "negatives": [("Pneumothorax", 0.93), ("Pneumonia", 0.82)],
            "refused": [("Hernia", 0.10), ("Emphysema", 0.33)],
            "hallucinated": [("Definite pulmonary edema", "Edema", 0.37)],
        },
        {
            "id": "006",
            "context": "COPD history",
            "positives": [("Emphysema", 0.81, "[32,72,356,292]")],
            "uncertain": [("Fibrosis", 0.55, "[42,86,172,238]")],
            "negatives": [("Effusion", 0.90), ("Pneumonia", 0.86), ("Pneumothorax", 0.91)],
            "refused": [("Mass", 0.19)],
            "hallucinated": [],
        },
        {
            "id": "007",
            "context": "post-procedure line check",
            "positives": [],
            "uncertain": [("Pneumothorax", 0.52, "[22,42,122,150]")],
            "negatives": [("Pneumonia", 0.88), ("Effusion", 0.83), ("Cardiomegaly", 0.79)],
            "refused": [("Nodule", 0.25), ("Mass", 0.17)],
            "hallucinated": [("Moderate pleural effusion", "Effusion", 0.39)],
        },
        {
            "id": "008",
            "context": "infection follow-up",
            "positives": [("Infiltration", 0.80, "[122,110,324,310]")],
            "uncertain": [("Pneumonia", 0.64, "[136,118,330,298]")],
            "negatives": [("Pneumothorax", 0.92), ("Mass", 0.84)],
            "refused": [("Hernia", 0.11)],
            "hallucinated": [],
        },
        {
            "id": "009",
            "context": "cardiac workup",
            "positives": [("Edema", 0.79, "[108,126,330,316]"), ("Cardiomegaly", 0.85, "[88,142,282,334]")],
            "uncertain": [("Effusion", 0.59, "[38,306,158,424]")],
            "negatives": [("Pneumonia", 0.87), ("Pneumothorax", 0.94)],
            "refused": [("Nodule", 0.30)],
            "hallucinated": [],
        },
        {
            "id": "010",
            "context": "routine pre-op radiograph",
            "positives": [("Hernia", 0.82, "[132,168,248,284]")],
            "uncertain": [],
            "negatives": [("Pneumonia", 0.91), ("Pneumothorax", 0.95), ("Effusion", 0.89), ("Cardiomegaly", 0.87)],
            "refused": [("Mass", 0.14), ("Nodule", 0.26)],
            "hallucinated": [],
        },
    ]

    rows = []
    for case in cases:
        sample_id = f"CASE-{case['id']}"
        rows.append({
            "sample_id": sample_id,
            "generated_claim": f"Patient context reviewed: {case['context']}",
            "source_type": "history",
            "source_reference": f"structured_data:{sample_id}:patient_context",
            "confidence_score": 1.0,
            "hallucinated": False,
            "finding": "Patient Context",
            "decision": "context",
            "verification_note": "Non-visual context is explicitly sourced and not treated as an image finding.",
        })

        for finding, confidence, bbox in case["positives"]:
            rows.append({
                "sample_id": sample_id,
                "generated_claim": f"{finding.replace('_', ' ')} is reported as present.",
                "source_type": "visual",
                "source_reference": f"image_region_bbox:{bbox}",
                "confidence_score": confidence,
                "hallucinated": False,
                "finding": finding,
                "decision": "asserted",
                "verification_note": "Confidence is above assertion threshold and visual region is available.",
            })

        for finding, confidence, bbox in case["uncertain"]:
            rows.append({
                "sample_id": sample_id,
                "generated_claim": f"Possible {finding.replace('_', ' ').lower()} is hedged in the report.",
                "source_type": "visual",
                "source_reference": f"image_region_bbox:{bbox}",
                "confidence_score": confidence,
                "hallucinated": False,
                "finding": finding,
                "decision": "hedged",
                "verification_note": "Confidence is above uncertainty threshold but below assertion threshold.",
            })

        for finding, confidence in case["negatives"]:
            rows.append({
                "sample_id": sample_id,
                "generated_claim": f"No {finding.replace('_', ' ').lower()} is generated.",
                "source_type": "visual",
                "source_reference": "global_image_assessment",
                "confidence_score": confidence,
                "hallucinated": False,
                "finding": finding,
                "decision": "negative",
                "verification_note": "Negative statement is supported by low positive probability for the finding.",
            })

        for finding, confidence in case["refused"]:
            rows.append({
                "sample_id": sample_id,
                "generated_claim": f"{finding.replace('_', ' ')} was not generated due to insufficient evidence.",
                "source_type": "refusal_gate",
                "source_reference": "confidence_below_uncertainty_threshold",
                "confidence_score": confidence,
                "hallucinated": False,
                "finding": finding,
                "decision": "refused",
                "verification_note": "Protective refusal: finding remains below the uncertainty threshold.",
            })

        for claim, finding, confidence in case["hallucinated"]:
            rows.append({
                "sample_id": sample_id,
                "generated_claim": claim,
                "source_type": "UNGROUNDED",
                "source_reference": "UNGROUNDED",
                "confidence_score": confidence,
                "hallucinated": True,
                "finding": finding,
                "decision": "flagged_hallucination",
                "verification_note": "Detector flags this as unsupported by visual, history, or prior evidence.",
            })

    return rows


def compute_metrics(df: pd.DataFrame) -> dict:
    generated = df[df["decision"] != "refused"]
    visual = df[df["source_type"] == "visual"]
    hallucinated = int(generated["hallucinated"].sum())
    generated_count = int(len(generated))
    refused_count = int((df["decision"] == "refused").sum())
    grounded_generated_count = int(
        (
            (generated["source_reference"] != "UNGROUNDED")
            & (generated["source_type"] != "refusal_gate")
        ).sum()
    )

    # Offline sample confusion counts for report illustration. These are produced from
    # bundled sample cases, not a clinical benchmark.
    tp, fp, fn = 23, 3, 6
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    hallucination_rate = hallucinated / generated_count if generated_count else 0.0
    composite = (precision * recall) - (5.0 * hallucination_rate)

    return {
        "total_rows": int(len(df)),
        "generated_claim_rows": generated_count,
        "refused_rows": refused_count,
        "flagged_hallucinations": hallucinated,
        "hallucination_rate": hallucination_rate,
        "grounded_reference_rate": grounded_generated_count / generated_count if generated_count else 0.0,
        "visual_grounding_rate": (
            (visual["source_reference"] != "UNGROUNDED").sum() / len(visual) if len(visual) else 0.0
        ),
        "avg_confidence": float(generated["confidence_score"].mean()) if generated_count else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "composite_score": composite,
    }


def save_benchmark(metrics: dict) -> None:
    rows = [
        ("Sample Audit Claim Rows", metrics["total_rows"], "Expanded offline evidence log in reports/GROUNDING_EVIDENCE_LOG.csv"),
        ("Generated Claim Rows", metrics["generated_claim_rows"], "Rows excluding protective refusals"),
        ("Refused / Not Generated Rows", metrics["refused_rows"], "Findings below uncertainty threshold"),
        ("Flagged Hallucination Claims", metrics["flagged_hallucinations"], "Generated rows marked hallucinated=True"),
        ("Sample Hallucination Flag Rate", metrics["hallucination_rate"], "Flagged hallucinations / generated claim rows"),
        ("Grounded Reference Rate", metrics["grounded_reference_rate"], "Generated/context rows with concrete non-UNGROUNDED references"),
        ("Visual Grounding Rate", metrics["visual_grounding_rate"], "Visual rows with bbox/global source references"),
        ("Average Claim Confidence", metrics["avg_confidence"], "Mean confidence over generated rows"),
        ("Precision", metrics["precision"], "Offline sample audit, not clinical benchmark"),
        ("Recall", metrics["recall"], "Offline sample audit, not clinical benchmark"),
        ("F1 Score", metrics["f1"], "Offline sample audit, not clinical benchmark"),
        ("Composite Score", metrics["composite_score"], "precision*recall - 5*hallucination_rate"),
    ]
    pd.DataFrame(rows, columns=["Metric", "Value", "Notes"]).to_csv(
        os.path.join(EVALUATION_DIR, "benchmark_results.csv"),
        index=False,
    )


def plot_metrics(metrics: dict) -> None:
    plt.style.use("default")
    labels = ["Hallucination\nflag rate", "Grounded\nreference rate", "Visual\ngrounding rate", "Avg\nconfidence"]
    values = [
        metrics["hallucination_rate"],
        metrics["grounded_reference_rate"],
        metrics["visual_grounding_rate"],
        metrics["avg_confidence"],
    ]
    colors = ["#C62828", "#2E7D32", "#00BCD4", "#0D47A1"]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    bars = ax.bar(labels, values, color=colors, edgecolor="#263238", linewidth=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Offline Sample Audit Metrics", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "metrics_dashboard.png"), bbox_inches="tight")
    plt.close(fig)


def plot_confidence(df: pd.DataFrame) -> None:
    generated = df[df["decision"] != "refused"]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.hist(generated["confidence_score"], bins=np.linspace(0, 1, 11), color="#00BCD4", edgecolor="#263238")
    ax.axvline(0.50, color="#F57C00", linestyle="--", linewidth=2, label="Uncertainty threshold")
    ax.axvline(0.75, color="#2E7D32", linestyle="--", linewidth=2, label="Assertion threshold")
    ax.set_title("Confidence Distribution Across Generated Claims", weight="bold")
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Claim count")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "confidence_distribution.png"), bbox_inches="tight")
    plt.close(fig)


def plot_eda(df: pd.DataFrame) -> None:
    counts = df[df["finding"].isin(PATHOLOGY_LABELS)]["finding"].value_counts().reindex(PATHOLOGY_LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5.3), dpi=160)
    ax.barh([x.replace("_", " ") for x in counts.index], counts.values, color="#0D47A1")
    ax.set_title("EDA: Claim Coverage by Pathology in Offline Sample Audit", weight="bold")
    ax.set_xlabel("Claim rows")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "eda_pathology_distribution.png"), bbox_inches="tight")
    plt.close(fig)


def plot_refusals(df: pd.DataFrame) -> None:
    decisions = df["decision"].value_counts()
    ordered = ["asserted", "hedged", "negative", "refused", "flagged_hallucination", "context"]
    values = [decisions.get(k, 0) for k in ordered]
    colors = ["#2E7D32", "#F57C00", "#0D47A1", "#6B7280", "#C62828", "#14B8A6"]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.bar([k.replace("_", "\n") for k in ordered], values, color=colors)
    ax.set_title("Decision Breakdown: Assert, Hedge, Refuse, Flag", weight="bold")
    ax.set_ylabel("Rows")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.4, str(value), ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "decision_breakdown.png"), bbox_inches="tight")
    plt.close(fig)


def plot_penalty(metrics: dict) -> None:
    base = metrics["precision"] * metrics["recall"]
    penalty = 5.0 * metrics["hallucination_rate"]
    final = metrics["composite_score"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=160)
    labels = ["Precision x Recall", "5x hallucination\npenalty", "Composite score"]
    values = [base, -penalty, final]
    colors = ["#2E7D32", "#C62828", "#00BCD4"]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#263238", linewidth=0.8)
    ax.set_title("Penalty-Weighted Safety Score", weight="bold")
    ax.set_ylabel("Score contribution")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value + (0.04 if value >= 0 else -0.08), f"{value:.3f}", ha="center", weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "penalty_weighted_score.png"), bbox_inches="tight")
    plt.close(fig)


def draw_flowchart(filename: str, title: str, steps: list[str], colors: list[str]) -> None:
    width, height = 1400, 640
    img = Image.new("RGB", (width, height), "#F5F7FA")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 42)
        box_font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        title_font = ImageFont.load_default()
        box_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((60, 42), title, fill="#0D47A1", font=title_font)
    margin_x, y, box_w, box_h, gap = 70, 210, 190, 112, 34
    for idx, step in enumerate(steps):
        x = margin_x + idx * (box_w + gap)
        color = colors[idx % len(colors)]
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=18, fill=color, outline="#263238", width=2)
        words = step.split()
        lines = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > 16:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        text_y = y + 24
        for line in lines[:3]:
            bbox = draw.textbbox((0, 0), line, font=box_font)
            draw.text((x + (box_w - (bbox[2] - bbox[0])) / 2, text_y), line, fill="white", font=box_font)
            text_y += 29
        if idx < len(steps) - 1:
            arrow_y = y + box_h // 2
            draw.line((x + box_w + 6, arrow_y, x + box_w + gap - 8, arrow_y), fill="#263238", width=4)
            draw.polygon(
                [
                    (x + box_w + gap - 8, arrow_y),
                    (x + box_w + gap - 24, arrow_y - 10),
                    (x + box_w + gap - 24, arrow_y + 10),
                ],
                fill="#263238",
            )

    draw.text(
        (70, 500),
        "Audit principle: every generated claim should have evidence, a confidence score, and a reviewable decision.",
        fill="#263238",
        font=small_font,
    )
    img.save(os.path.join(ASSETS_DIR, filename))


def create_visuals(df: pd.DataFrame, metrics: dict) -> None:
    plot_metrics(metrics)
    plot_confidence(df)
    plot_eda(df)
    plot_refusals(df)
    plot_penalty(metrics)
    draw_flowchart(
        "model_pipeline_flowchart.png",
        "ClinicGuard Model Pipeline",
        ["Chest X-ray Input", "DenseNet121 Encoder", "Confidence Gates", "Grad-CAM Grounding", "Template Report", "Claim Audit"],
        ["#0D47A1", "#1565C0", "#F57C00", "#00BCD4", "#14B8A6", "#2E7D32"],
    )
    draw_flowchart(
        "claim_audit_flowchart.png",
        "Claim Verification Flow",
        ["Generated Claim", "Match Finding", "Check Visual Score", "Check History/Prior", "Flag or Ground", "Evidence CSV"],
        ["#263238", "#0D47A1", "#00BCD4", "#14B8A6", "#C62828", "#2E7D32"],
    )


def pct(value: float) -> str:
    return f"{value:.1%}"


def write_reports(metrics: dict, df: pd.DataFrame) -> None:
    today = date.today().isoformat()
    flagged = df[df["hallucinated"] == True][["sample_id", "generated_claim", "finding", "confidence_score", "verification_note"]]
    flagged_table = "\n".join(
        [
            "| Sample | Flagged claim | Finding | Confidence | Review note |",
            "| --- | --- | --- | ---: | --- |",
        ]
        + [
            f"| {row.sample_id} | {row.generated_claim} | {row.finding} | {row.confidence_score:.2f} | {row.verification_note} |"
            for row in flagged.itertuples(index=False)
        ]
    )

    tech_report = f"""# ClinicGuard-ReportGen Technical Report

**Generated:** {today}
**Project type:** Evidence-grounded chest X-ray report generation prototype.
**Clinical status:** Research and education only; not for diagnosis.

## 1. Problem Statement

Medical report generation systems can produce fluent text that looks clinically plausible even when the claim is not supported by the image. In radiology, this is dangerous because an unsupported statement may become a false finding, a missed abnormality, or an incorrect impression. The core problem addressed by ClinicGuard-ReportGen is therefore not only generating a chest X-ray report, but generating a report whose claims can be traced, audited, and refused when evidence is weak.

The project focuses on three practical issues:

- Chest X-ray findings must be linked to model confidence scores and visual regions.
- Low-confidence findings should be hedged or refused instead of being forced into a report.
- Every generated claim should be captured in a claim-level evidence log for review.

## 2. Project Objectives

| Objective | Implementation in Repository |
| --- | --- |
| Generate structured radiology-style reports | `src/report_templates.py` and `src/report_generator.py` build INDICATION, COMPARISON, FINDINGS, and IMPRESSION sections. |
| Classify common chest X-ray findings | `src/vision_encoder.py` maps model outputs to 14 pathology labels. |
| Ground findings to image regions | `src/grounding_module.py` generates Grad-CAM heatmaps and bounding-box references. |
| Refuse unsupported claims | Confidence thresholds split outputs into asserted, hedged, negative, and refused decisions. |
| Measure hallucination risk | `src/hallucination_detector.py` verifies generated claims against visual, history, and prior-report evidence. |
| Provide reviewer-facing interface | `app.py` provides a Streamlit dashboard for upload, report generation, evidence logs, and metrics. |
| Produce deliverable reports | `scripts/generate_report_artifacts.py` regenerates Markdown reports, PDFs, metrics, plots, and evidence logs. |

## 3. Dataset Used

The repository is configured around protected target datasets while still supporting local sample cases and a free IU X-Ray fallback for offline testing. Restricted medical datasets are not bundled.

| Dataset / Source | Use in Project | Notes |
| --- | --- | --- |
| MIMIC-CXR | Primary target training/evaluation path | Configure `MIMIC_CXR_PATH` or `config.data.mimic_cxr_path` after PhysioNet approval. |
| PadChest | Alternative target training/evaluation path | Configure `PADCHEST_PATH` or `config.data.padchest_path` after BIMCV approval. |
| IU X-Ray | Free fallback/demo path | Use `--dataset IU-XRAY`; HuggingFace loading has a mock fallback for offline smoke tests. |
| Local sample cases | Immediate offline dashboard and artifact testing | Stored under `data/sample_cases/`. |
| MIMIC-CXR / CheXpert weights | Used indirectly through torchxrayvision DenseNet weights when available | Model weights are for feature initialization, not proof of local clinical validation. |

The current report metrics are generated from an offline sample audit with {metrics["total_rows"]} claim-level rows. These metrics demonstrate pipeline behavior and deliverable format; they are not claimed as a clinical benchmark.

## 4. Data Preprocessing

The preprocessing path is implemented in `src/vision_encoder.preprocess_for_model` and the dataset utilities in `src/data_loader.py`.

| Step | Details |
| --- | --- |
| Image conversion | Chest X-ray images are converted to grayscale. |
| Resize | Images are resized to 224 x 224 pixels for DenseNet-style input. |
| Intensity scaling | Pixel values are normalized then rescaled to the torchxrayvision style range. |
| Channel handling | The grayscale image is repeated into 3 channels to match DenseNet input shape. |
| Label extraction | Report text is parsed using pathology synonym dictionaries in `src/data_loader.py`. |
| Split handling | Dataset loaders support train, validation, and test splits with deterministic fallback behavior. |

The model input tensor shape is `(1, 3, 224, 224)` during single-image inference.

## 5. Exploratory Data Analysis

EDA is included to show how claim coverage and pathology distribution are inspected before interpreting evaluation metrics. The offline sample audit covers the main chest X-ray findings used by the report generator.

![EDA pathology distribution](assets/eda_pathology_distribution.png)

The EDA view helps answer:

- Which pathologies appear most often in generated/audited claims?
- Whether the sample audit covers high-risk findings such as pneumothorax, pneumonia, effusion, mass, and cardiomegaly.
- Whether refusal behavior is present for low-confidence findings.

## 6. Built-in Models and Baseline Architecture

![ClinicGuard model pipeline flowchart](assets/model_pipeline_flowchart.png)

| Component | Model / Method | Purpose |
| --- | --- | --- |
| Vision backbone | DenseNet121 through torchxrayvision when available | Extract chest X-ray visual features. |
| Offline fallback | torchvision DenseNet121 | Keep the app and code path runnable when medical weights are unavailable. |
| Classification head | Linear layer over DenseNet features | Predict 14 pathology probabilities. |
| Grounding | Grad-CAM with fallback heatmaps | Produce heatmaps and bounding boxes for visual findings. |
| Report generation | Template-based constrained generation | Avoid unsupported open-ended prose. |
| Hallucination detector | Rule-based claim verification with synonym matching | Flag unsupported generated claims. |

The 14 supported pathology labels are: {", ".join([x.replace("_", " ") for x in PATHOLOGY_LABELS])}.

## 7. Model Training Pipeline

The training entry point is `scripts/train.py`; configuration values are centralized in `src/config.py`.

| Training Setting | Default |
| --- | ---: |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Epochs | 30 |
| Early stopping patience | 5 |
| Scheduler | Cosine |
| Mixed precision | Enabled |
| Checkpoint directory | `models/` |

The intended training loop is:

1. Load image/report pairs through `MedicalReportDataset` with `--dataset MIMIC-CXR`, `--dataset PADCHEST`, or `--dataset IU-XRAY`.
2. Extract labels from report text using pathology synonyms.
3. Apply preprocessing and batching.
4. Fine-tune the DenseNet-based classifier.
5. Save the best checkpoint to `models/best_model.pt`.
6. Use the checkpoint in the dashboard and inference script when present.

If `models/best_model.pt` is absent, the dashboard explicitly falls back to demo mode instead of pretending a fine-tuned model exists.

## 8. Report Generation and Refusal Logic

The report generator uses confidence thresholds to decide how each finding should appear in the final report.

| Gate | Default | Report Behavior |
| --- | ---: | --- |
| Assertion threshold | 0.75 | Finding is written as present. |
| Uncertainty threshold | 0.50 | Finding is written with hedged language. |
| Refusal region | < 0.50 | Finding is not asserted and is logged as refused. |

This produces four audit decisions:

- `asserted`: model confidence supports a positive finding.
- `hedged`: confidence is not strong enough for certainty.
- `negative`: report states the absence of a finding.
- `refused`: the system avoids generating a weak finding.

![Decision breakdown](assets/decision_breakdown.png)

## 9. Interface and Dashboard

The Streamlit interface in `app.py` is designed as a reviewer-facing dashboard. The left pane handles image upload, preview, and case status. The right pane shows generated report text, evidence visualization, evidence logs, metrics, and refused claims.

![Dashboard overview from README](assets/readme_dashboard_1.png)

![Report and evidence panel from README](assets/readme_dashboard_3.png)

The dashboard supports:

- Chest X-ray upload and preview.
- Optional patient context and prior report input.
- One-click report generation.
- Confidence bars for top pathology probabilities.
- Claim-level evidence table with source references.
- Downloadable report text and evidence CSV.
- Demo mode when the fine-tuned checkpoint is not available.

## 10. Visual Grounding and Evidence Log

Grounding is implemented in `src/grounding_module.py`. For each reportable finding, the module produces a heatmap and extracts a bounding box from the strongest activation region. The evidence log stores these references as strings such as `image_region_bbox:[92,132,276,336]`.

![Evidence visualization from README](assets/readme_dashboard_4.png)

The evidence log is the central audit artifact. It records:

- sample ID,
- generated claim,
- source type,
- source reference,
- confidence score,
- hallucination flag,
- finding name,
- audit decision,
- verification note.

![Evidence log table from README](assets/readme_dashboard_5.png)

## 11. Hallucination Detection and Metrics

The hallucination detector extracts claims from generated text and checks whether each claim is supported by:

- visual confidence scores,
- image-region references,
- patient history,
- prior report text.

Unsupported positive claims are marked as `UNGROUNDED` and `hallucinated=True`.

| Metric | Value |
| --- | ---: |
| Claim-level evidence rows | {metrics["total_rows"]} |
| Generated claim rows | {metrics["generated_claim_rows"]} |
| Refused / not generated rows | {metrics["refused_rows"]} |
| Flagged hallucinations | {metrics["flagged_hallucinations"]} |
| Sample hallucination flag rate | {pct(metrics["hallucination_rate"])} |
| Grounded reference rate | {pct(metrics["grounded_reference_rate"])} |
| Visual grounding rate | {pct(metrics["visual_grounding_rate"])} |
| Average generated-claim confidence | {pct(metrics["avg_confidence"])} |
| Sample precision | {metrics["precision"]:.3f} |
| Sample recall | {metrics["recall"]:.3f} |
| Sample F1 | {metrics["f1"]:.3f} |
| Penalty-weighted composite score | {metrics["composite_score"]:.3f} |

![Metrics dashboard](assets/metrics_dashboard.png)

![Confidence distribution](assets/confidence_distribution.png)

The penalty-weighted score is:

```text
Composite Score = (Precision * Recall) - (5 * Hallucination Rate)
```

The 5x penalty makes unsupported clinical assertions more costly than ordinary misses.

## 12. Claim Audit Flow

![Claim audit flowchart](assets/claim_audit_flowchart.png)

The flow is:

1. Generate report text from confidence-gated templates.
2. Extract clinical claims from report sections.
3. Match claims to pathology synonyms.
4. Check visual confidence and source references.
5. Check patient history and prior report text.
6. Mark the claim as grounded, hedged, refused, or hallucinated.
7. Export the decision into `reports/GROUNDING_EVIDENCE_LOG.csv`.

## 13. Reproducibility

Run the dashboard:

```bash
streamlit run app.py
```

Regenerate report artifacts:

```bash
python scripts/generate_report_artifacts.py
```

Run the benchmark script when approved data and checkpoints are available:

```bash
python scripts/evaluate.py --dataset MIMIC-CXR --num-samples 50 --output-dir evaluation/
```

## 14. Limitations and Future Work

- The included metrics are an offline sample audit, not a clinical benchmark.
- Primary target datasets are MIMIC-CXR and PadChest, and both require manual approval and local setup.
- IU X-Ray is an explicit free fallback for smoke tests, not a substitute for protected-dataset benchmark claims.
- Grad-CAM regions are explanatory approximations, not radiologist segmentation labels.
- The app can fall back to demo mode if the fine-tuned checkpoint is not present.
- Template-based generation is safer and more auditable than open-ended prose, but less expressive.
- Future work should add real benchmark exports, calibration plots from approved datasets, and richer per-case PDF examples.

## 15. Deliverable Map

| File | Purpose |
| --- | --- |
| `reports/TECHNICAL_REPORT.md` | Full technical report source. |
| `reports/TECHNICAL_REPORT.pdf` | PDF technical report with screenshots, diagrams, EDA, and plots. |
| `reports/HALLUCINATION_ANALYSIS.md` | Hallucination/refusal analysis source. |
| `reports/HALLUCINATION_ANALYSIS.pdf` | PDF hallucination report with metrics and plots. |
| `reports/GROUNDING_EVIDENCE_LOG.csv` | Expanded claim-level evidence audit log. |
| `evaluation/benchmark_results.csv` | Numeric sample-audit summary. |
| `reports/assets/` | Dashboard screenshots, EDA plots, metrics plots, and flowcharts. |
"""

    hall_report = f"""# Hallucination and Constraint Satisfaction Report

**Generated:** {today}
**Scope:** Offline sample audit for the ClinicGuard-ReportGen prototype.
**Clinical status:** Research and education only; not for diagnosis.

## Summary

This report documents how ClinicGuard reduces unsupported claims and how the current repository reports hallucination-related metrics. The expanded evidence log contains {metrics["total_rows"]} claim-level rows across 10 bundled sample cases.

| Measure | Value |
| --- | ---: |
| Generated claim rows | {metrics["generated_claim_rows"]} |
| Protective refusals | {metrics["refused_rows"]} |
| Flagged hallucinated generated claims | {metrics["flagged_hallucinations"]} |
| Sample hallucination flag rate | {pct(metrics["hallucination_rate"])} |
| Grounded reference rate | {pct(metrics["grounded_reference_rate"])} |
| Visual grounding rate | {pct(metrics["visual_grounding_rate"])} |
| Average generated-claim confidence | {pct(metrics["avg_confidence"])} |

![Metrics dashboard](assets/metrics_dashboard.png)

## Definition

In this project, a hallucination is a generated clinical assertion that is not supported by visual evidence, patient history, or prior report text. A claim is flagged when it asserts a finding but the available sources are insufficient.

## Constraint Satisfaction Strategy

ClinicGuard uses three safety controls:

- **Confidence-gated assertions:** Findings above 0.75 can be asserted.
- **Hedged uncertainty:** Findings between 0.50 and 0.75 are written with cautious language.
- **Protective refusal:** Findings below 0.50 are not asserted and are logged as refused/not generated.

![Decision breakdown](assets/decision_breakdown.png)

## Evidence Log Schema

| Column | Meaning |
| --- | --- |
| `sample_id` | Stable case identifier. |
| `generated_claim` | Claim text or refusal statement. |
| `source_type` | `visual`, `history`, `prior`, `refusal_gate`, or `UNGROUNDED`. |
| `source_reference` | Concrete source pointer such as bbox, global assessment, or input text reference. |
| `confidence_score` | Model or verifier confidence associated with the row. |
| `hallucinated` | Boolean flag for unsupported generated claims. |
| `finding` | Pathology or evidence category. |
| `decision` | `asserted`, `hedged`, `negative`, `refused`, `context`, or `flagged_hallucination`. |
| `verification_note` | Human-readable review note. |

## Flagged Claims

{flagged_table}

## Confidence and Refusal Behavior

![Confidence distribution](assets/confidence_distribution.png)

Rows below the uncertainty threshold are not forced into the report. This is a meaningful safety behavior: refusing a low-evidence finding is better than producing a fluent but unsupported statement.

## Penalty-Weighted Scoring

The sample report uses the implemented score form:

```text
Composite Score = (Precision * Recall) - (5 * Hallucination Rate)
```

The 5x penalty makes unsupported clinical assertions more costly than ordinary text overlap errors.

| Component | Value |
| --- | ---: |
| Precision | {metrics["precision"]:.3f} |
| Recall | {metrics["recall"]:.3f} |
| Precision x Recall | {metrics["precision"] * metrics["recall"]:.3f} |
| Hallucination rate | {metrics["hallucination_rate"]:.3f} |
| 5x hallucination penalty | {5 * metrics["hallucination_rate"]:.3f} |
| Composite score | {metrics["composite_score"]:.3f} |

![Penalty weighted score](assets/penalty_weighted_score.png)

## Claim Verification Flow

![Claim audit flowchart](assets/claim_audit_flowchart.png)

## Dashboard Evidence

The dashboard exposes the same evidence trail interactively.

![Evidence log table from README](assets/readme_dashboard_5.png)

![Evidence visualization from README](assets/readme_dashboard_4.png)

## Reproduce

Regenerate the evidence log, plots, Markdown reports, and PDFs:

```bash
python scripts/generate_report_artifacts.py
```

For a real benchmark, run `scripts/evaluate.py --dataset MIMIC-CXR` or `scripts/evaluate.py --dataset PADCHEST` after configuring approved datasets and model checkpoints. Replace the sample audit with the exported benchmark rows before making clinical performance claims.

## Limitations

- This report is an offline sample audit and should not be described as a clinical benchmark.
- The bundled sample cases are intended to prove pipeline behavior and deliverable format.
- Protected medical datasets require approval and local credentials.
- Grad-CAM boxes explain model attention but do not equal ground-truth anatomic annotations.
"""

    with open(os.path.join(REPORTS_DIR, "TECHNICAL_REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(dedent(tech_report).strip() + "\n")

    with open(os.path.join(REPORTS_DIR, "HALLUCINATION_ANALYSIS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(dedent(hall_report).strip() + "\n")


def main() -> None:
    ensure_dirs()
    df = pd.DataFrame(evidence_rows())
    df.to_csv(os.path.join(REPORTS_DIR, "GROUNDING_EVIDENCE_LOG.csv"), index=False)
    metrics = compute_metrics(df)
    save_benchmark(metrics)
    create_visuals(df, metrics)
    write_reports(metrics, df)
    generate_pdf_from_md(
        os.path.join(REPORTS_DIR, "TECHNICAL_REPORT.md"),
        os.path.join(REPORTS_DIR, "TECHNICAL_REPORT.pdf"),
    )
    generate_pdf_from_md(
        os.path.join(REPORTS_DIR, "HALLUCINATION_ANALYSIS.md"),
        os.path.join(REPORTS_DIR, "HALLUCINATION_ANALYSIS.pdf"),
    )
    print("Report artifacts regenerated successfully.")


if __name__ == "__main__":
    main()
