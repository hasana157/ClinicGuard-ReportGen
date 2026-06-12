"""
CLI script to acquire and prepare chest X-ray datasets.

Supports downloading the Indiana University Chest X-Ray (IU X-Ray) dataset from HuggingFace,
and displays instructions for credentialed access datasets (MIMIC-CXR, PadChest).
"""

import os
import sys
import argparse
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ensure_directories, RAW_DATA_DIR, PROCESSED_DATA_DIR


def download_iu_xray(save_dir: str):
    """
    Downloads the IU X-Ray dataset from HuggingFace and saves details locally.

    Args:
        save_dir: Path where raw dataset should be stored.
    """
    print("==============================================================")
    print("📥 Downloading IU X-Ray (Indiana University Chest X-Ray) Dataset")
    print("==============================================================")
    
    try:
        from datasets import load_dataset
        print("Connecting to HuggingFace dataset hub...")
        
        # Load datasets (using pre-paired version ykumards/open-i or dz-osamu/IU-Xray)
        print("Downloading 'dz-osamu/IU-Xray' dataset. Please wait...")
        dataset = load_dataset("dz-osamu/IU-Xray", split="train")
        
        print(f"✅ Download completed! Successfully fetched {len(dataset)} samples.")
        
        # Save placeholder info/metadata for reference
        os.makedirs(save_dir, exist_ok=True)
        meta_file = os.path.join(save_dir, "iu_xray_metadata.csv")
        
        # Save a subset of metadata as CSV to show data structure in data/raw
        records = []
        for i in range(min(100, len(dataset))):
            item = dataset[i]
            records.append({
                "id": f"IU-{i}",
                "findings": item.get("findings", ""),
                "impression": item.get("impression", ""),
            })
        
        import pandas as pd
        df = pd.DataFrame(records)
        df.to_csv(meta_file, index=False)
        print(f"Saved metadata sample file to: {meta_file}")
        
    except ImportError:
        print("❌ Error: 'datasets' package is not installed.")
        print("Please run: pip install datasets")
        print("Generating mock data folder structure instead.")
        _generate_mock_directories(save_dir)
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("Generating mock data folder structure to proceed offline.")
        _generate_mock_directories(save_dir)


def _generate_mock_directories(save_dir: str):
    """Generate dummy directories and files for offline runs."""
    os.makedirs(save_dir, exist_ok=True)
    meta_file = os.path.join(save_dir, "iu_xray_metadata.csv")
    with open(meta_file, "w") as f:
        f.write("id,findings,impression\n")
        f.write("MOCK-001,\"Heart size is normal. Lungs are clear. No effusion.\",\"No acute abnormality.\"\n")
        f.write("MOCK-002,\"Cardiomegaly is present. Bilateral pleural effusions.\",\"Heart failure.\"\n")
    print(f"Created mock metadata file at: {meta_file}")


def print_mimic_cxr_instructions():
    """Prints step-by-step instructions to register and request MIMIC-CXR."""
    print("==============================================================")
    print("🔑 Instructions for Acquiring MIMIC-CXR Dataset (PhysioNet)")
    print("==============================================================")
    print("MIMIC-CXR contains protected health information (PHI) and requires")
    print("credentialed access. Please follow these steps:")
    print("1. Create an account on PhysioNet: https://physionet.org/register/")
    print("2. Complete the required CITI training course:")
    print("   'Data or Specimens Only Research' (under Human Subjects Research)")
    print("3. Submit your training certificate to PhysioNet for credentialing.")
    print("4. Sign the Data Use Agreement (DUA) for MIMIC-CXR:")
    print("   https://physionet.org/content/mimic-cxr-jpg/2.0.0/")
    print("5. Once approved, download the images and reports using the physionet-cli:")
    print("   physionet-cli download physionet.org/content/mimic-cxr-jpg/2.0.0/ -d ./data/raw/mimic-cxr")
    print("==============================================================")


def print_padchest_instructions():
    """Prints instructions for PadChest dataset."""
    print("==============================================================")
    print("🔑 Instructions for Acquiring PadChest Dataset (BIMCV)")
    print("==============================================================")
    print("PadChest requires approval from the BIMCV organization.")
    print("1. Visit the BIMCV project portal: http://bimcv.cipf.es/bimcv-projects/padchest/")
    print("2. Fill out the application form requesting access.")
    print("3. Once the request is approved, you will receive download link instructions.")
    print("4. Download the dataset (~1TB) or the Grounded Reporting subset (PadChest-GR, ~5GB)")
    print("   and extract to: ./data/raw/padchest")
    print("==============================================================")


def main():
    parser = argparse.ArgumentParser(description="Dataset downloader utility")
    parser.add_argument("--iu-xray", action="store_true", help="Download the free IU X-Ray dataset")
    parser.add_argument("--mimic", action="store_true", help="Print instructions for MIMIC-CXR")
    parser.add_argument("--padchest", action="store_true", help="Print instructions for PadChest")
    parser.add_argument("--all", action="store_true", help="Trigger all actions")
    
    args = parser.parse_args()
    
    # Ensure project directories exist
    ensure_directories()
    
    # If no flags are provided, show help
    if not (args.iu-xray or args.mimic or args.padchest or args.all):
        parser.print_help()
        print("\nDefaulting to downloading free IU X-Ray dataset:")
        download_iu_xray(RAW_DATA_DIR)
        return

    if args.all or args.iu-xray:
        download_iu_xray(RAW_DATA_DIR)
        
    if args.all or args.mimic:
        print_mimic_cxr_instructions()
        
    if args.all or args.padchest:
        print_padchest_instructions()


if __name__ == "__main__":
    main()
