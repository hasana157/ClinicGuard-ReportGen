"""
Utility script to generate sample cases (chest X-rays, patient histories, prior reports)
for end-to-end testing and demo execution.
"""

import os
import json
import numpy as np
from PIL import Image

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SAMPLE_CASES_DIR, ensure_directories


def main():
    # Ensure directories exist
    ensure_directories()
    
    print(f"Generating sample cases in: {SAMPLE_CASES_DIR}")
    
    # 1. Generate Mock Images
    # We will generate a few mock chest X-ray images (grayscale, simulating rib cages/lungs)
    for i in range(1, 6):
        img_path = os.path.join(SAMPLE_CASES_DIR, f"chest_xray_{i:03d}.png")
        if not os.path.exists(img_path):
            # Create a 256x256 image with simulated lungs (two darker circles)
            img_arr = np.ones((256, 256), dtype=np.uint8) * 180
            
            # Left Lung
            for y in range(60, 200):
                for x in range(40, 110):
                    dist = ((y - 130) ** 2) / (70 ** 2) + ((x - 75) ** 2) / (35 ** 2)
                    if dist <= 1.0:
                        img_arr[y, x] = int(60 + 100 * dist)
                        
            # Right Lung
            for y in range(60, 200):
                for x in range(146, 216):
                    dist = ((y - 130) ** 2) / (70 ** 2) + ((x - 181) ** 2) / (35 ** 2)
                    if dist <= 1.0:
                        img_arr[y, x] = int(60 + 100 * dist)
                        
            # Heart shadow (overlapping center/left lung)
            for y in range(120, 195):
                for x in range(95, 155):
                    dist = ((y - 160) ** 2) / (40 ** 2) + ((x - 125) ** 2) / (30 ** 2)
                    if dist <= 1.0:
                        img_arr[y, x] = int(220 - 40 * dist)
                        
            # Rib lines (horizontal stripes)
            for y in range(70, 200, 25):
                img_arr[y:y+3, 30:226] = 210
                
            img = Image.fromarray(img_arr).convert("RGB")
            img.save(img_path)
            print(f"  Created sample image: {img_path}")
            
    # Also save a default 'chest_xray.png' for README quick start
    default_img_path = os.path.join(SAMPLE_CASES_DIR, "chest_xray.png")
    if not os.path.exists(default_img_path):
        img = Image.open(os.path.join(SAMPLE_CASES_DIR, "chest_xray_001.png"))
        img.save(default_img_path)
        print(f"  Created default sample image: {default_img_path}")

    # 2. Generate Patient History JSON files
    histories = [
        {"history": "65-year-old male presenting with acute shortness of breath and chest pain. History of chronic hypertension.", "patient_id": "X123"},
        {"history": "45-year-old female with persistent cough and fever for 5 days. Suspicion of pneumonia.", "patient_id": "Y456"},
        {"history": "72-year-old male with chronic COPD and worsening dyspnea.", "patient_id": "Z789"},
        {"history": "28-year-old female status post motor vehicle collision, chest wall tenderness.", "patient_id": "W987"},
        {"history": "Routine pre-operative screening chest radiograph. No acute cardiopulmonary complaints.", "patient_id": "V321"}
    ]
    
    for i, hist in enumerate(histories):
        hist_path = os.path.join(SAMPLE_CASES_DIR, f"patient_history_{i+1:03d}.json")
        with open(hist_path, "w") as f:
            json.dump(hist, f, indent=4)
        print(f"  Created sample history: {hist_path}")
        
    # Default patient_history.json
    default_hist_path = os.path.join(SAMPLE_CASES_DIR, "patient_history.json")
    with open(default_hist_path, "w") as f:
        json.dump(histories[0], f, indent=4)
        
    # 3. Generate Prior Reports
    priors = [
        "Chest radiograph dated 2026-06-08: Mild cardiomegaly and trace bilateral pleural effusions. Lungs otherwise clear.",
        "Chest radiograph dated 2026-05-15: Right lower lobe consolidation representing pneumonia. Heart size within normal limits.",
        "Chest radiograph dated 2026-04-20: Emphysematous changes and hyperinflation. No acute pulmonary infiltrate.",
        "Chest radiograph dated 2026-02-12: Normal study. Clear lung fields. Heart size normal.",
        "None available for comparison."
    ]
    
    for i, prior in enumerate(priors):
        prior_path = os.path.join(SAMPLE_CASES_DIR, f"prior_report_{i+1:03d}.txt")
        with open(prior_path, "w") as f:
            f.write(prior)
        print(f"  Created sample prior: {prior_path}")
        
    # Default prior_report.txt
    default_prior_path = os.path.join(SAMPLE_CASES_DIR, "prior_report.txt")
    with open(default_prior_path, "w") as f:
        f.write(priors[0])
        
    print("Success: Completed generating sample cases.")


if __name__ == "__main__":
    main()
