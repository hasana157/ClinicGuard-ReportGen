"""
Inference script.

Generates radiology reports and Grad-CAM visualizations on input images (single image
or directory of images) and writes evidence logs to CSV.
"""

import os
import sys
import json
import argparse
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config, ProjectConfig
from src.vision_encoder import GroundedVisionEncoder
from src.grounding_module import GradCAMGrounding
from src.report_generator import ConstrainedReportGenerator, ReportOutput


def run_inference(
    image_path: str,
    generator: ConstrainedReportGenerator,
    patient_history: Optional[str] = None,
    prior_report: Optional[str] = None,
    sample_id: str = "001"
) -> ReportOutput:
    """
    Run end-to-end inference on a single image.

    Args:
        image_path: Path to the image.
        generator: ConstrainedReportGenerator instance.
        patient_history: Optional clinical details.
        prior_report: Optional prior report.
        sample_id: Sample identifier.

    Returns:
        ReportOutput object.
    """
    # Load original image
    image = Image.open(image_path).convert("RGB")
    
    # Run report generation pipeline
    output = generator.generate(
        image=image,
        patient_history=patient_history,
        prior_report=prior_report,
        sample_id=sample_id
    )
    
    return output


def main():
    parser = argparse.ArgumentParser(description="End-to-end report generation inference")
    parser.add_argument("--image", type=str, required=True, help="Path to single image or directory of images")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint .pt file")
    parser.add_argument("--history", type=str, default=None, help="Patient clinical history text or path to JSON")
    parser.add_argument("--prior", type=str, default=None, help="Prior report text or path to txt file")
    parser.add_argument("--output", type=str, default="./results", help="Directory to save generated outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    config = get_config()
    device = torch.device(args.device)
    
    # 1. Load model
    model = GroundedVisionEncoder(config=config.vision)
    model.to(device)
    
    if args.model and os.path.exists(args.model):
        print(f"Loading checkpoint from: {args.model}")
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    else:
        print("Using pre-trained torchxrayvision weights (no checkpoint specified).")
        
    model.eval()
    
    # 2. Setup modules
    grounding = GradCAMGrounding(model, config=config.gradcam)
    generator = ConstrainedReportGenerator(model, grounding, config=config.report)
    
    # 3. Resolve inputs
    patient_history = None
    if args.history:
        if os.path.exists(args.history):
            with open(args.history, "r") as f:
                if args.history.endswith(".json"):
                    data = json.load(f)
                    patient_history = data.get("history", data.get("patient_history", ""))
                else:
                    patient_history = f.read().strip()
        else:
            patient_history = args.history

    prior_report = None
    if args.prior:
        if os.path.exists(args.prior):
            with open(args.prior, "r") as f:
                prior_report = f.read().strip()
        else:
            prior_report = args.prior

    os.makedirs(args.output, exist_ok=True)
    
    # 4. Handle files
    image_paths = []
    if os.path.isdir(args.image):
        for f in os.listdir(args.image):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(args.image, f))
    else:
        image_paths.append(args.image)
        
    if not image_paths:
        print(f"No valid images found in: {args.image}")
        return
        
    print(f"Processing {len(image_paths)} images...")
    
    all_evidence = []
    
    for idx, img_path in enumerate(tqdm(image_paths)):
        name = os.path.splitext(os.path.basename(img_path))[0]
        sample_id = f"S-{idx+1:03d}"
        
        try:
            output = run_inference(
                image_path=img_path,
                generator=generator,
                patient_history=patient_history,
                prior_report=prior_report,
                sample_id=sample_id
            )
            
            # Save Report Text
            report_out_path = os.path.join(args.output, f"{name}_report.txt")
            with open(report_out_path, "w") as f:
                f.write(output.report_text)
                
            # Save Grad-CAM visualizations
            for g_res in output.grounding_results:
                if g_res.visualization:
                    vis_path = os.path.join(args.output, f"{name}_grounding_{g_res.finding}.png")
                    g_res.visualization.save(vis_path)
                    
            # Accumulate evidence
            all_evidence.extend(output.evidence_log)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    # Save cumulative evidence log CSV
    if all_evidence:
        evidence_df = pd.DataFrame(all_evidence)
        csv_path = os.path.join(args.output, "grounding_evidence_log.csv")
        evidence_df.to_csv(csv_path, index=False)
        print(f"✅ Generated grounding evidence log (CSV) with {len(evidence_df)} rows saved to: {csv_path}")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
