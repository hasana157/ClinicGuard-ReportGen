# Setup

ClinicGuard ReportGen supports three dataset modes:

| Dataset | Purpose | Access |
|---|---|---|
| `MIMIC-CXR` | Primary target dataset | PhysioNet credentialing required |
| `PADCHEST` | Alternative target dataset | BIMCV approval required |
| `IU-XRAY` | Free fallback/demo dataset | HuggingFace dataset |

## 1. Install Dependencies

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Configure Datasets

Edit `src/config.py` or set environment variables:

```bash
set MIMIC_CXR_PATH=D:\datasets\mimic-cxr-jpg
set PADCHEST_PATH=D:\datasets\padchest
set CLINICGUARD_PRIMARY_DATASET=MIMIC-CXR
```

The default local paths are:

```text
data/raw/mimic-cxr
data/raw/padchest
```

## 3. MIMIC-CXR

1. Register at PhysioNet.
2. Complete required credentialing and CITI training.
3. Request access to MIMIC-CXR-JPG.
4. Download and extract the dataset locally.
5. Validate the local copy:

```bash
python scripts/download_datasets.py --mimic D:\datasets\mimic-cxr-jpg
```

Expected structure:

```text
mimic-cxr/
  files/pXX/pXXXXXXXX/sXXXXXXXX/*.jpg
  mimic-cxr-2.0.0-metadata.csv
  mimic-cxr-2.0.0-chexpert.csv
```

Reports may be provided as `mimic-cxr-reports.csv` or as `sXXXXXXXX.txt` files in the MIMIC-CXR file tree.

## 4. PadChest

1. Request access from the BIMCV PadChest project.
2. Download and extract the approved dataset locally.
3. Validate the local copy:

```bash
python scripts/download_datasets.py --padchest D:\datasets\padchest
```

Expected structure:

```text
padchest/
  PADCHEST_chest_x_ray_images_labels_*.csv
  image_dir/*.png
```

The loader also checks common alternatives such as `images/`.

## 5. IU X-Ray Fallback

Use IU X-Ray only for development, smoke tests, and fallback demonstrations:

```bash
python scripts/download_datasets.py --iu-xray
python scripts/train.py --dataset IU-XRAY --sample-limit 25 --epochs 1
python scripts/evaluate.py --dataset IU-XRAY --num-samples 10
```

## 6. Train and Evaluate

```bash
python scripts/download_datasets.py --check-all
python scripts/train.py --dataset MIMIC-CXR --epochs 15
python scripts/evaluate.py --dataset MIMIC-CXR --num-samples 100
```

The included evidence log and PDFs are offline sample-audit artifacts. Regenerate reports after training/evaluating on approved local datasets before claiming dataset-specific benchmark metrics.
