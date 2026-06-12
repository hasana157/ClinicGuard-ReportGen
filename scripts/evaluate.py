"""
Evaluation script to benchmark the system on the test dataset.

Runs the GroundedVisionEncoder and ConstrainedReportGenerator on the test split,
computes classification and text quality metrics, generates benchmark_results.csv and
GROUNDING_EVIDENCE_LOG.csv, and prints a final metrics table.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config, ProjectConfig, PATHOLOGY_LABELS
from src.data_loader import IUXRayDataset
from src.vision_encoder import GroundedVisionEncoder
from src.grounding_module import GradCAMGrounding
from src.report_generator import ConstrainedReportGenerator
from src.evaluation import MedicalReportEvaluator


def main():
    parser = argparse.ArgumentParser(description="System Evaluation & Benchmarking script")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./evaluation", help="Directory to save evaluation results")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of test samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    config = get_config()
    device = torch.device(args.device)
    
    # 1. Load model and config
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
    evaluator = MedicalReportEvaluator(config=config.evaluation)
    
    # 3. Load test dataset
    print("Loading test dataset splits...")
    # Evaluate transforms
    from src.preprocessing import get_eval_transforms
    eval_transform = get_eval_transforms(config.vision.input_size)
    
    test_dataset = IUXRayDataset(split="test", transform=eval_transform, config=config)
    print(f"Loaded {len(test_dataset)} test samples.")
    
    num_samples = len(test_dataset)
    if args.num_samples is not None:
        num_samples = min(args.num_samples, num_samples)
        print(f"Limiting evaluation to first {num_samples} samples.")

    # 4. Generate Predictions & Reports
    print("Running inference pipeline across test set...")
    
    generated_reports = []
    ground_truth_reports = []
    
    all_pred_labels = []
    all_true_labels = []
    
    all_evidence_logs = []
    
    sample_outputs_dir = os.path.join(args.output_dir, "sample_outputs")
    os.makedirs(sample_outputs_dir, exist_ok=True)
    
    for idx in range(num_samples):
        sample = test_dataset[idx]
        image_tensor = sample["image"].unsqueeze(0)  # Add batch dim
        true_labels = sample["labels"].numpy()
        
        # Load image PIL from dataset if available, else convert tensor to PIL
        # (This is just for visual grounding visualizations)
        from src.data_loader import extract_labels_from_report
        # Recreate PIL from tensor or just use a blank one for speed if original PIL not saved
        img_np = (sample["image"].squeeze(0).numpy() + 1024.0) / 2048.0
        img_np = (img_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(img_np).convert("RGB")
        
        # Generate report
        sample_id = f"TEST-{idx+1:03d}"
        output = generator.generate(
            image=image_pil, 
            sample_id=sample_id
        )
        
        generated_reports.append(output.report_text)
        ground_truth_reports.append(sample["report"])
        
        # Binary predictions
        pred_probs = output.confidence_scores
        pred_binary = [1.0 if pred_probs[label] >= config.report.evidence_threshold else 0.0 for label in PATHOLOGY_LABELS]
        
        all_pred_labels.append(pred_binary)
        # Binarize ground truth (-1 uncertainty -> 0)
        true_binary = [1.0 if val == 1.0 else 0.0 for val in true_labels]
        all_true_labels.append(true_binary)
        
        # Evidence log
        all_evidence_logs.extend(output.evidence_log)
        
        # Save a few sample outputs (first 10)
        if idx < 10:
            sample_report_path = os.path.join(sample_outputs_dir, f"{sample_id}_generated_report.txt")
            with open(sample_report_path, "w") as f:
                f.write(output.report_text)
                
            # Save visual grounding figures
            for g_res in output.grounding_results:
                if g_res.visualization:
                    vis_path = os.path.join(sample_outputs_dir, f"{sample_id}_grounding_{g_res.finding}.png")
                    g_res.visualization.save(vis_path)

    # 5. Evaluate Metrics
    print("Computing metrics...")
    results = evaluator.evaluate(
        predicted_labels=np.array(all_pred_labels),
        true_labels=np.array(all_true_labels),
        generated_reports=generated_reports,
        ground_truth_reports=ground_truth_reports,
        evidence_logs=all_evidence_logs
    )
    
    # 6. Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print results summary table
    report_table = evaluator.generate_report_table(results)
    print(report_table)
    
    summary_path = os.path.join(args.output_dir, "benchmark_results.csv")
    summary_data = {
        "Metric": [
            "Hallucination Rate", "Grounding Success Rate", "Precision", 
            "Recall", "F1 Score", "Composite Score", "BLEU", 
            "ROUGE-1", "ROUGE-2", "ROUGE-L"
        ],
        "Value": [
            results.hallucination_rate, results.grounding_success, results.precision,
            results.recall, results.f1, results.composite_score, results.bleu,
            results.rouge1, results.rouge2, results.rougeL
        ]
    }
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
    print(f"Saved benchmark summary to: {summary_path}")
    
    # Save GROUNDING_EVIDENCE_LOG.csv
    reports_dir = os.path.join(os.path.dirname(args.output_dir), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    evidence_path = os.path.join(reports_dir, "GROUNDING_EVIDENCE_LOG.csv")
    pd.DataFrame(all_evidence_logs).to_csv(evidence_path, index=False)
    print(f"Saved grounding evidence log with {len(all_evidence_logs)} rows to: {evidence_path}")


if __name__ == "__main__":
    main()
