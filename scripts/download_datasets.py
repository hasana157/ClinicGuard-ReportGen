"""
Dataset acquisition and validation utility.

MIMIC-CXR and PadChest require external approvals, so this script validates
local dataset structure instead of pretending to download restricted data.
IU X-Ray remains available as a free HuggingFace fallback for demos.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RAW_DATA_DIR, ProjectConfig, ensure_directories


def validate_mimic_cxr(path: str) -> bool:
    """Validate a local MIMIC-CXR-JPG style directory."""
    root = Path(path)
    print(f"Checking MIMIC-CXR at: {root}")

    if not root.exists():
        print("ERROR: path does not exist.")
        print("Download MIMIC-CXR-JPG after PhysioNet approval:")
        print("  https://physionet.org/content/mimic-cxr-jpg/2.0.0/")
        return False

    files_dir = root / "files"
    metadata = _first_existing(
        root,
        [
            "mimic-cxr-2.0.0-metadata.csv",
            "mimic-cxr-metadata.csv",
            "metadata.csv",
        ],
        ["*metadata*.csv"],
    )
    reports = _first_existing(
        root,
        ["mimic-cxr-reports.csv", "mimic-cxr-2.0.0-reports.csv", "reports.csv"],
        ["*reports*.csv", "*report*.csv"],
    )
    labels = _first_existing(
        root,
        ["mimic-cxr-2.0.0-chexpert.csv", "chexpert.csv"],
        ["*chexpert*.csv", "*negbio*.csv"],
    )

    ok = files_dir.exists() and (metadata is not None or _has_images(files_dir))
    if ok:
        print("OK: MIMIC-CXR image tree found.")
        if metadata:
            print(f"OK: metadata CSV: {metadata.name}")
        if reports:
            print(f"OK: report CSV: {reports.name}")
        elif _has_text_reports(files_dir):
            print("OK: text reports found in files/ tree.")
        else:
            print("WARN: no report CSV/text reports found; labels will be used where available.")
        if labels:
            print(f"OK: label CSV: {labels.name}")
        print(f"Sample image count check: {_count_matches(files_dir, ['*.jpg', '*.jpeg', '*.png'])}")
        return True

    print("ERROR: MIMIC-CXR structure incomplete.")
    print("Expected at minimum:")
    print("  mimic-cxr/")
    print("    files/pXX/pXXXXXXXX/sXXXXXXXX/*.jpg")
    print("    mimic-cxr-2.0.0-metadata.csv  (recommended)")
    print("    mimic-cxr-2.0.0-chexpert.csv   (recommended)")
    return False


def validate_padchest(path: str) -> bool:
    """Validate a local PadChest directory."""
    root = Path(path)
    print(f"Checking PadChest at: {root}")

    if not root.exists():
        print("ERROR: path does not exist.")
        print("Request PadChest access from:")
        print("  http://bimcv.cipf.es/bimcv-projects/padchest/")
        return False

    metadata = _first_existing(
        root,
        [
            "PADCHEST_chest_x_ray_images_labels_01_v2.csv",
            "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
            "padchest.csv",
        ],
        ["PADCHEST*labels*.csv", "*padchest*.csv", "*.csv"],
    )
    image_dirs = [root / "image_dir", root / "images", root / "Images", root]
    image_count = sum(_count_matches(d, ["*.png", "*.jpg", "*.jpeg"]) for d in image_dirs if d.exists())

    if metadata and image_count > 0:
        print(f"OK: metadata CSV: {metadata.name}")
        print(f"Sample image count check: {image_count}")
        return True

    print("ERROR: PadChest structure incomplete.")
    print("Expected at minimum:")
    print("  padchest/")
    print("    PADCHEST_chest_x_ray_images_labels_*.csv")
    print("    image_dir/*.png  (or images/*.png)")
    return False


def download_iu_xray(save_dir: str) -> bool:
    """Fetch a small IU X-Ray metadata sample from HuggingFace."""
    print("Downloading/checking IU X-Ray from HuggingFace...")
    try:
        from datasets import load_dataset
        import pandas as pd

        dataset = load_dataset("dz-osamu/IU-Xray", split="train")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        metadata_path = Path(save_dir) / "iu_xray_metadata.csv"
        rows = []
        for idx in range(min(100, len(dataset))):
            item = dataset[idx]
            rows.append(
                {
                    "id": f"IU-{idx}",
                    "findings": item.get("findings", ""),
                    "impression": item.get("impression", ""),
                }
            )
        pd.DataFrame(rows).to_csv(metadata_path, index=False)
        print(f"OK: fetched {len(dataset)} IU X-Ray samples.")
        print(f"Saved metadata sample to: {metadata_path}")
        return True
    except ImportError:
        print("ERROR: install the datasets package first: pip install datasets")
    except Exception as exc:
        print(f"ERROR: IU X-Ray download/check failed: {exc}")
    return False


def print_access_instructions() -> None:
    """Print concise protected-dataset access instructions."""
    print("\nProtected dataset access:")
    print("MIMIC-CXR:")
    print("  1. Register at https://physionet.org/register/")
    print("  2. Complete required CITI training and credentialing.")
    print("  3. Request access to https://physionet.org/content/mimic-cxr-jpg/2.0.0/")
    print("  4. Download locally, then set MIMIC_CXR_PATH or config.data.mimic_cxr_path.")
    print("PadChest:")
    print("  1. Request access at http://bimcv.cipf.es/bimcv-projects/padchest/")
    print("  2. Extract locally, then set PADCHEST_PATH or config.data.padchest_path.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or fetch configured datasets")
    parser.add_argument("--check-all", action="store_true", help="Validate configured MIMIC-CXR and PadChest paths")
    parser.add_argument("--mimic", nargs="?", const="", help="Validate MIMIC-CXR path; omit value to use config path")
    parser.add_argument("--padchest", nargs="?", const="", help="Validate PadChest path; omit value to use config path")
    parser.add_argument("--iu-xray", action="store_true", help="Download/check the free IU X-Ray fallback")
    parser.add_argument("--instructions", action="store_true", help="Print protected dataset access instructions")

    args = parser.parse_args()
    ensure_directories()
    config = ProjectConfig()

    ran = False
    ok = True

    if args.check_all:
        ran = True
        ok = validate_mimic_cxr(config.data.mimic_cxr_path) and ok
        ok = validate_padchest(config.data.padchest_path) and ok

    if args.mimic is not None:
        ran = True
        path = args.mimic or config.data.mimic_cxr_path
        ok = validate_mimic_cxr(path) and ok

    if args.padchest is not None:
        ran = True
        path = args.padchest or config.data.padchest_path
        ok = validate_padchest(path) and ok

    if args.iu_xray:
        ran = True
        ok = download_iu_xray(RAW_DATA_DIR) and ok

    if args.instructions:
        ran = True
        print_access_instructions()

    if not ran:
        parser.print_help()
        print_access_instructions()
        return

    sys.exit(0 if ok else 1)


def _first_existing(root: Path, filenames: Iterable[str], patterns: Iterable[str]) -> Optional[Path]:
    for filename in filenames:
        candidate = root / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    for pattern in patterns:
        matches = sorted(p for p in root.glob(pattern) if p.is_file())
        if matches:
            return matches[0]
    return None


def _count_matches(root: Path, patterns: Iterable[str], limit: int = 100000) -> int:
    count = 0
    for pattern in patterns:
        for _ in root.rglob(pattern):
            count += 1
            if count >= limit:
                return count
    return count


def _has_images(root: Path) -> bool:
    return _count_matches(root, ["*.jpg", "*.jpeg", "*.png"], limit=1) > 0


def _has_text_reports(root: Path) -> bool:
    return _count_matches(root, ["*.txt"], limit=1) > 0


if __name__ == "__main__":
    main()
