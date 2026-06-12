# ClinicGuard ReportGen

**Evidence-grounded radiology report generation with claim-level verification and source traceability**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research Prototype](https://img.shields.io/badge/status-Research%20Prototype-orange.svg)](#limitations)

---

## Overview

ClinicGuard ReportGen addresses a critical challenge in AI-assisted radiology: **hallucinated findings**. Traditional language models generate free-form reports that may contain unsupported claims. This prototype implements a controlled, evidence-based approach to radiology report generation.

By combining vision encoding, confidence-gated templates, visual grounding, and claim verification, ClinicGuard produces reports where every statement is:
- ✓ Linked to supporting evidence (image regions, patient history, prior findings)
- ✓ Backed by model confidence thresholds
- ✓ Recorded in an auditable evidence log
- ✓ Verifiable claim-by-claim

---

## Important: Dataset Configuration

The training and evaluation pipeline is designed for three dataset modes:

| Dataset | Role | Access |
|---|---|---|
| `MIMIC-CXR` | Primary target dataset | Requires PhysioNet credentialing and local download |
| `PADCHEST` | Alternative target dataset | Requires BIMCV approval and local download |
| `IU-XRAY` | Free fallback/demo dataset | HuggingFace download, with mock fallback for offline smoke tests |

The default configured primary dataset is `MIMIC-CXR`. If you do not have protected dataset access yet, run commands with `--dataset IU-XRAY` and keep the included sample-audit reports labeled as demonstrations, not clinical benchmark results.

Configure local protected dataset paths in `src/config.py` or with environment variables:

```bash
set MIMIC_CXR_PATH=D:\datasets\mimic-cxr-jpg
set PADCHEST_PATH=D:\datasets\padchest
```

Validate local datasets before training:

```bash
python scripts/download_datasets.py --check-all
python scripts/download_datasets.py --mimic D:\datasets\mimic-cxr-jpg
python scripts/download_datasets.py --padchest D:\datasets\padchest
```

---

## Key Features

### 🔗 Evidence Traceability
- Every generated claim references its source: image bounding boxes, patient history, or prior report text
- CSV evidence logs enable claim-level auditing
- Grad-CAM heatmaps visualize the image regions supporting each finding

### 🚫 Controlled Refusal
- Findings below confidence thresholds are rejected or hedged with uncertainty qualifiers
- Prevents high-risk unsupported statements
- Configurable thresholds for different finding types

### 📊 Claim Verification
- Hallucination detection module checks consistency between visual evidence and text
- Identifies claims that lack sufficient grounding
- Produces validation metadata for each report

### 🎯 Template-Based Generation
- Constrained report structure prevents unconstrained free-text hallucination
- Fills evidence-verified claims into structured templates
- Maintains clinical report format expectations

### 📈 Comprehensive Evaluation
- Penalty-weighted evaluation metrics for hallucination sensitivity
- Reproducible benchmark framework
- Sample evaluation artifacts included

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- 2 GB free disk space (sample data)
- CUDA-capable GPU recommended (CPU mode supported)

### Installation

```bash
# Clone repository
git clone https://github.com/hasana157/ClinicGuard-ReportGen.git
cd ClinicGuard-ReportGen

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run the Interactive Dashboard

```bash
streamlit run app.py
```

The Streamlit dashboard provides:
- **Image Upload & Display**: Load chest X-rays for analysis
- **Real-time Report Generation**: Watch claims generated with confidence scores
- **Evidence Visualization**: View Grad-CAM heatmaps and bounding boxes
- **Evidence Log**: Inspect the claim-level audit trail
- **Patient Context**: Optional patient history and prior report integration

**Dashboard Screenshots:**

> **[Dashboard Hero - Image Upload & Controls]**  
<img width="2479" height="1129" alt="image" src="https://github.com/user-attachments/assets/cc526eb8-f5c4-426e-97c1-0a1b6820abb6" />

<img width="2089" height="1091" alt="image" src="https://github.com/user-attachments/assets/94b25652-021c-422f-81fe-46d92040e7b0" />




> **[Report Generation Panel - Live Claims & Confidence]**  
<img width="1578" height="1016" alt="image" src="https://github.com/user-attachments/assets/61602936-e25e-4eb2-84f5-9d36d2d68976" />


> **[Evidence Visualization - Heatmaps & Overlays]**  
<img width="688" height="986" alt="image" src="https://github.com/user-attachments/assets/4ab2875a-10ce-4109-868e-de89fd4c4927" />



> **[Evidence Log Table - Audit Trail]**  
<img width="741" height="837" alt="image" src="https://github.com/user-attachments/assets/8ecc90bf-d233-4d34-9a1f-c4d3f3bddb3d" />


### Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates local sample chest X-rays and patient metadata for offline testing.

### Run Inference on a Single Case

```bash
python scripts/inference.py \
  --image data/sample_cases/chest_xray.png \
  --history data/sample_cases/patient_history.json \
  --prior data/sample_cases/prior_report.txt \
  --output results/
```

Output includes:
- `generated_report.txt` – Full radiology report
- `evidence_log.csv` – Claim-level grounding
- `grounding_overlay.png` – Visualized evidence regions

### Train a Model

```bash
python scripts/train.py \
  --epochs 10 \
  --batch-size 8 \
  --lr 1e-4 \
  --dataset MIMIC-CXR
```

Use `--dataset IU-XRAY` for the free fallback when MIMIC-CXR/PadChest are not available locally.

### Evaluate on Benchmark

```bash
python scripts/evaluate.py \
  --dataset MIMIC-CXR \
  --num-samples 10 \
  --output-dir evaluation/
```

Produces:
- `evaluation/benchmark_results.csv` – Metrics across samples
- `reports/GROUNDING_EVIDENCE_LOG.csv` – Complete claim audit trail
- `evaluation/sample_outputs/` – Generated reports and visualizations

### Regenerate Technical Report Artifacts

```bash
python scripts/generate_report_artifacts.py
```

Produces:
- `reports/TECHNICAL_REPORT.pdf` – Technical model/pipeline report with dashboard screenshots, EDA, metrics plots, and diagrams
- `reports/HALLUCINATION_ANALYSIS.pdf` – Constraint satisfaction and hallucination analysis report
- `reports/assets/` – Local dashboard screenshots, EDA plots, flowcharts, and metric figures
- `reports/GROUNDING_EVIDENCE_LOG.csv` – Expanded offline sample audit log
- `evaluation/benchmark_results.csv` – Numeric sample-audit summary

---

## Architecture

### Pipeline Overview

```
Chest X-ray Image
      ↓
┌─────────────────────────────────────┐
│  DenseNet121 Vision Encoder         │ → Pathology confidence scores
│  (Feature extraction & classification)
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Confidence Gating & Uncertainty    │ → Refusal/uncertainty decisions
│  (MC dropout uncertainty quantif.)   │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Grad-CAM Visual Grounding          │ → Localized evidence regions
│  (Interpretability & heatmaps)      │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Template-Based Report Builder      │ → Structured claim generation
│  (+ optional patient history/prior)  │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Hallucination Detector             │ → Verification scores
│  (Claim consistency checker)         │
└─────────────────────────────────────┘
      ↓
Report + Evidence Log (CSV)
- Generated findings
- Confidence scores
- Evidence sources
- Hallucination flags
```

### Core Components

| Module | Purpose |
|--------|---------|
| `vision_encoder.py` | DenseNet121 feature extraction and pathology classification |
| `grounding_module.py` | Grad-CAM heatmap generation and bounding box extraction |
| `report_generator.py` | Confidence-gated template filling and claim generation |
| `hallucination_detector.py` | Claim consistency verification against evidence |
| `uncertainty_quantifier.py` | Monte Carlo dropout-based confidence estimation |
| `evaluation.py` | Benchmark suite with penalty-weighted metrics |

---

## Project Structure

```
ClinicGuard-ReportGen/
├── app.py                          # Streamlit interactive dashboard
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation metadata
│
├── src/
│   ├── config.py                   # Paths, pathology labels, thresholds
│   ├── data_loader.py              # MIMIC-CXR, PadChest, and IU X-Ray loaders
│   ├── preprocessing.py            # Image and report preprocessing
│   ├── vision_encoder.py           # DenseNet121 feature extractor
│   ├── grounding_module.py         # Grad-CAM and bounding boxes
│   ├── report_generator.py         # Constrained generation pipeline
│   ├── report_templates.py         # Structured report templates
│   ├── hallucination_detector.py   # Claim verification
│   ├── uncertainty_quantifier.py   # MC dropout helpers
│   └── evaluation.py               # Evaluation suite
│
├── scripts/
│   ├── download_datasets.py        # Dataset validation and IU X-Ray fallback
│   ├── generate_sample_data.py     # Local sample data generation
│   ├── train.py                    # Training CLI
│   ├── inference.py                # Inference CLI
│   ├── evaluate.py                 # Evaluation CLI
|   |-- generate_pdfs.py            # Markdown-to-PDF report compiler
|   \-- generate_report_artifacts.py # Evidence log, plots, reports, and PDFs
│
├── data/sample_cases/              # Local sample X-rays and metadata
├── evaluation/                     # Benchmark artifacts
├── reports/                        # Technical documentation, PDFs, and assets
├── notebooks/                      # Exploratory analysis
└── docs/
    └── SETUP.md                    # Dataset setup and validation
```

---

## Evidence Log Format

Each generated claim is recorded in CSV format with full grounding metadata:

### Schema

```csv
sample_id,generated_claim,source_type,source_reference,confidence_score,hallucinated,finding,decision,verification_note
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | string | Unique identifier for the case |
| `generated_claim` | string | The generated text claim (e.g., "Cardiomegaly present") |
| `source_type` | enum | One of: `visual`, `history`, `prior`, `refusal_gate`, or `UNGROUNDED` |
| `source_reference` | string | Pointer to evidence source (e.g., `image_region_bbox:[100,150,250,300]`, `global_image_assessment`, or `confidence_below_uncertainty_threshold`) |
| `confidence_score` | float | Model confidence on [0.0, 1.0] |
| `hallucinated` | boolean | Flagged as inconsistent with source evidence |
| `finding` | string | Pathology or context category associated with the row |
| `decision` | enum | One of: `asserted`, `hedged`, `negative`, `refused`, `context`, or `flagged_hallucination` |
| `verification_note` | string | Human-readable reason for the audit decision |

### Example

```csv
sample_id,generated_claim,source_type,source_reference,confidence_score,hallucinated,finding,decision,verification_note
CASE-001,Cardiomegaly is reported as present.,visual,image_region_bbox:[92,132,276,336],0.94,False,Cardiomegaly,asserted,Confidence is above assertion threshold and visual region is available.
CASE-001,Possible edema is hedged in the report.,visual,image_region_bbox:[118,152,322,302],0.58,False,Edema,hedged,Confidence is above uncertainty threshold but below assertion threshold.
CASE-001,Mass was not generated due to insufficient evidence.,refusal_gate,confidence_below_uncertainty_threshold,0.21,False,Mass,refused,Protective refusal: finding remains below the uncertainty threshold.
CASE-001,Large right upper lobe mass,UNGROUNDED,UNGROUNDED,0.21,True,Mass,flagged_hallucination,Detector flags this as unsupported by visual history or prior evidence.
```

---

## Datasets

### Local Sample Data (Included)

Pre-bundled sample cases are included for dashboard and artifact smoke tests.

```bash
python scripts/generate_sample_data.py
```

### MIMIC-CXR (Primary Target)

MIMIC-CXR requires PhysioNet credentialing and a local download. After approval, place the dataset under `data/raw/mimic-cxr` or set `MIMIC_CXR_PATH`.

```bash
python scripts/download_datasets.py --mimic D:\datasets\mimic-cxr-jpg
python scripts/train.py --dataset MIMIC-CXR --epochs 15
python scripts/evaluate.py --dataset MIMIC-CXR --num-samples 100
```

### PadChest (Alternative Target)

PadChest requires BIMCV approval. After extraction, place the dataset under `data/raw/padchest` or set `PADCHEST_PATH`.

```bash
python scripts/download_datasets.py --padchest D:\datasets\padchest
python scripts/train.py --dataset PADCHEST --epochs 15
python scripts/evaluate.py --dataset PADCHEST --num-samples 100
```

### IU X-Ray (Free Fallback)

Use IU X-Ray only when protected datasets are unavailable. This is useful for smoke tests and demonstration runs.

```bash
python scripts/download_datasets.py --iu-xray
python scripts/train.py --dataset IU-XRAY --epochs 10
python scripts/evaluate.py --dataset IU-XRAY --num-samples 10
```

The included reports and evidence log are offline sample-audit artifacts unless you regenerate them after configuring an approved dataset and model checkpoint.

---

## Configuration

### Dataset Selection

Edit `src/config.py` or set environment variables before running training/evaluation:

```python
PRIMARY_DATASET = "MIMIC-CXR"
MIMIC_CXR_PATH = "./data/raw/mimic-cxr"
PADCHEST_PATH = "./data/raw/padchest"
IU_XRAY_HF_DATASET = "dz-osamu/IU-Xray"
```

Command-line flags override the configured primary dataset:

```bash
python scripts/train.py --dataset MIMIC-CXR
python scripts/evaluate.py --dataset PADCHEST
python scripts/train.py --dataset IU-XRAY --sample-limit 25
```

### Key Thresholds

Edit `src/config.py` to adjust report generation behavior:

```python
confidence_threshold_positive = 0.75
confidence_threshold_uncertain = 0.50
confidence_threshold_refusal = 0.50
hallucination_penalty_weight = 5.0
```

---

## Usage Examples

### Example 1: Standalone Report Generation

```python
from src.vision_encoder import DenseNetEncoder
from src.report_generator import ReportGenerator
from PIL import Image

# Load image
image = Image.open("chest_xray.png")

# Generate report with evidence
encoder = DenseNetEncoder()
generator = ReportGenerator()

report, evidence_log = generator.generate(
    image=image,
    patient_history={"age": 65, "smoking": "yes"},
    prior_report=None
)

# Save outputs
report.to_file("report.txt")
evidence_log.to_csv("evidence.csv", index=False)
```

### Example 2: Batch Evaluation

```python
from src.evaluation import BenchmarkEvaluator

evaluator = BenchmarkEvaluator(
    num_samples=100,
    penalize_hallucination=True,
    hallucination_weight=2.0
)

results = evaluator.run()
results.to_csv("benchmark.csv")
```

### Example 3: Custom Confidence Thresholds

```python
from src.config import CONFIDENCE_THRESHOLDS

# Strict mode for critical findings
CONFIDENCE_THRESHOLDS['critical'] = 0.95
CONFIDENCE_THRESHOLDS['major'] = 0.85

# Re-run inference with stricter criteria
```

---

## Evaluation Metrics

The system measures:

- **Hallucination Rate**: Percentage of claims flagged as unsupported
- **Recall (Evidence)**: What fraction of true findings are captured
- **Precision (Grounding)**: What fraction of generated claims have valid evidence
- **Confidence Calibration**: Do confidence scores reflect actual accuracy?
- **Penalty-Weighted Loss**: Clinical-grade metric emphasizing high-risk hallucinations

See `src/evaluation.py` for metric definitions.

---

## Limitations

### Clinical Context
- **Not a diagnostic system**: This is a workflow prototype for evidence-based report generation, not a validated clinical tool
- **Grad-CAM localization** is an interpretability aid, not expert-grade segmentation
- **Confidence scores** reflect model uncertainty, not clinical probability
- **Sample data** is for offline development; real data requires institutional review

### Technical Scope
- Primary configured dataset is `MIMIC-CXR`, but protected datasets require external approvals and local paths
- `IU-XRAY` is available as an explicit free fallback for development and smoke tests
- Model checkpoints are not bundled; train locally or provide `--model` during evaluation
- GPU strongly recommended; CPU inference is slow

### Evidence Log Caveats
- Included `reports/GROUNDING_EVIDENCE_LOG.csv` contains an expanded offline sample audit log with claim decisions, refusals, and hallucination flags
- Included PDFs and plots demonstrate the reporting format and audit pipeline
- Do not describe included metrics as MIMIC-CXR or PadChest benchmark results unless regenerated with approved local datasets and checkpoints

### Known Issues
- Grad-CAM heatmaps may highlight background regions on low-contrast images
- Report templates assume AP/PA chest X-ray views; lateral or portable views may reduce accuracy
- MC dropout uncertainty estimates require multiple forward passes (~100 ms overhead)

---

## Contributing

We welcome feedback and contributions. Please note:

1. **Issue Reports**: Describe the problem, expected vs. actual output, and environment
2. **Pull Requests**: Include tests and documentation updates
3. **Research Extensions**: Open a discussion issue before implementing major changes

---

## Citation

If you use this project in research, please cite:

```bibtex
@software{clinicguard_2024,
  author = {[Author Name(s)]},
  title = {ClinicGuard ReportGen: Evidence-Grounded Radiology Report Generation},
  url = {https://github.com/hasana157/ClinicGuard-ReportGen},
  year = {2024},
  note = {Research prototype. Not for clinical use.}
}
```

---

## License

MIT License – see [LICENSE](LICENSE) for details.

---

## Support & Questions

- **Issues & Bugs**: [GitHub Issues](https://github.com/hasana157/ClinicGuard-ReportGen/issues)
- **Documentation**: See `docs/` directory
- **Discussions**: [GitHub Discussions](https://github.com/hasana157/ClinicGuard-ReportGen/discussions)

---

## Acknowledgments

- DenseNet121 architecture from [torchvision](https://pytorch.org/vision/)
- Grad-CAM methodology from [Selvaraju et al. (2017)](https://arxiv.org/abs/1610.02055)
- IU X-Ray dataset: [Indiana University School of Medicine](https://r2r.cis.upenn.edu/cxr/)

---

