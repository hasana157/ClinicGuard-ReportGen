"""
Constrained report generation engine.

Executes the zero-hallucination report generation pipeline: runs the vision encoder,
applies the confidence-based refusal mechanism, triggers visual grounding per finding,
builds the structured report, compiles the evidence log (CSV formatting), and returns
the final report outputs.
"""

import os
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from src.config import ReportGeneratorConfig, PATHOLOGY_LABELS, get_config
from src.vision_encoder import preprocess_for_model
from src.grounding_module import GradCAMGrounding, GroundingResult
from src.report_templates import build_full_report


@dataclass
class ReportOutput:
    """Dataclass containing the outputs of the generation pipeline."""
    report_text: str
    findings: List[str]
    impression: str
    evidence_log: List[Dict[str, Any]]
    grounding_results: List[GroundingResult]
    confidence_scores: Dict[str, float]
    hallucination_flags: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConstrainedReportGenerator:
    """Constrained report generation engine with refusal gates and evidence tracking."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        grounding_module: GradCAMGrounding,
        config: Optional[ReportGeneratorConfig] = None
    ):
        """
        Initialize the report generator.

        Args:
            encoder: PyTorch vision encoder.
            grounding_module: GradCAM grounding module.
            config: Report generator configuration.
        """
        self.encoder = encoder
        self.grounding_module = grounding_module
        self.config = config if config else ReportGeneratorConfig()
        self.device = next(encoder.parameters()).device

    def generate(
        self,
        image: Image.Image,
        patient_history: Optional[str] = None,
        prior_report: Optional[str] = None,
        sample_id: str = "001"
    ) -> ReportOutput:
        """
        Generate a radiology report for an input image.

        Args:
            image: PIL Image object.
            patient_history: Patient clinical context.
            prior_report: Prior report context.
            sample_id: Identifier for logging.

        Returns:
            ReportOutput dataclass.
        """
        # Preprocess image
        image_tensor = preprocess_for_model(image).to(self.device)
        
        # Run classification model
        with torch.no_grad():
            classifier_out = self.encoder.classify(
                image_tensor, 
                threshold=self.config.uncertainty_threshold
            )
            
        predictions = classifier_out["confidences"]
        
        # Apply Refusal Mechanism
        # Keep only findings that are above uncertainty threshold (0.50)
        # Any findings below this are treated as 0.0 (abstain / refuse to mention)
        filtered_predictions = {}
        refused_findings = []
        for finding, conf in predictions.items():
            if conf >= self.config.uncertainty_threshold:
                filtered_predictions[finding] = conf
            else:
                filtered_predictions[finding] = 0.0
                refused_findings.append(finding)
                
        # Generate Grad-CAM Visual Grounding for positive & uncertain findings
        grounding_results = self.grounding_module.ground_all_findings(
            image_tensor=image_tensor,
            original_image=image,
            predictions=filtered_predictions,
            threshold=self.config.uncertainty_threshold
        )
        
        # Build Report Text
        report_text = build_full_report(
            predictions=filtered_predictions,
            patient_history=patient_history,
            prior_report=prior_report,
            threshold_positive=self.config.evidence_threshold,
            threshold_uncertain=self.config.uncertainty_threshold
        )
        
        # Assemble Evidence Log
        evidence_log = self._build_evidence_log(
            sample_id=sample_id,
            predictions=filtered_predictions,
            grounding_results=grounding_results,
            patient_history=patient_history,
            prior_report=prior_report
        )
        
        # Import detector inside method to avoid circular imports
        from src.hallucination_detector import HallucinationDetector
        detector = HallucinationDetector()
        
        # Post-generation factuality verification
        hallucination_flags = detector.detect_hallucinations(
            report_text=report_text,
            evidence={
                "visual": filtered_predictions,
                "history": patient_history,
                "prior": prior_report,
                "grounding": {r.finding: r.bbox for r in grounding_results}
            }
        )
        
        # Annotate evidence log with hallucination flags
        for entry in evidence_log:
            claim = entry["generated_claim"]
            # Check if this claim was flagged as hallucination
            is_hallucinated = False
            for flag in hallucination_flags:
                # If claim matches or is contained
                if flag["claim"].lower() in claim.lower() or claim.lower() in flag["claim"].lower():
                    is_hallucinated = flag["hallucinated"]
                    break
            entry["hallucinated"] = is_hallucinated

        # Extract findings and impression
        findings = []
        impression = []
        
        findings_started = False
        impression_started = False
        for line in report_text.split("\n"):
            line = line.strip()
            if line.startswith("FINDINGS:"):
                findings_started = True
                impression_started = False
                continue
            elif line.startswith("IMPRESSION:"):
                findings_started = False
                impression_started = True
                continue
                
            if findings_started and line:
                findings.append(line)
            elif impression_started and line:
                impression.append(line)
                
        findings_str = "\n".join(findings)
        impression_str = "\n".join(impression)

        return ReportOutput(
            report_text=report_text,
            findings=findings,
            impression=impression,
            evidence_log=evidence_log,
            grounding_results=grounding_results,
            confidence_scores=predictions,
            hallucination_flags=hallucination_flags,
            metadata={"refused_findings": refused_findings}
        )

    def _build_evidence_log(
        self,
        sample_id: str,
        predictions: Dict[str, float],
        grounding_results: List[GroundingResult],
        patient_history: Optional[str] = None,
        prior_report: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Builds the grounding evidence log mapping claims to source references.
        """
        log = []
        
        # 1. Log patient history source if provided
        if patient_history:
            log.append({
                "sample_id": sample_id,
                "generated_claim": f"Patient history: {patient_history}",
                "source_type": "history",
                "source_reference": "structured_data:patient_history_input",
                "confidence_score": 1.0,
                "hallucinated": False
            })
            
        # 2. Log prior report source if provided
        if prior_report:
            log.append({
                "sample_id": sample_id,
                "generated_claim": f"Prior report: {prior_report}",
                "source_type": "prior",
                "source_reference": "text_data:prior_report_input",
                "confidence_score": 1.0,
                "hallucinated": False
            })

        # Map grounding results
        grounding_map = {r.finding: r for r in grounding_results}
        
        # Check active findings
        for finding, conf in predictions.items():
            if conf >= self.config.uncertainty_threshold:
                # Get visual bbox reference
                g_res = grounding_map.get(finding)
                if g_res:
                    bbox_str = f"image_region_bbox:[{g_res.bbox['x1']},{g_res.bbox['y1']},{g_res.bbox['x2']},{g_res.bbox['y2']}]"
                else:
                    bbox_str = "global_image_assessment"
                    
                log.append({
                    "sample_id": sample_id,
                    "generated_claim": f"Presence of {finding.replace('_', ' ')}",
                    "source_type": "visual",
                    "source_reference": bbox_str,
                    "confidence_score": conf,
                    "hallucinated": False
                })
            else:
                # Log negative assessments / normals
                log.append({
                    "sample_id": sample_id,
                    "generated_claim": f"Absence of {finding.replace('_', ' ')}",
                    "source_type": "visual",
                    "source_reference": "global_image_assessment",
                    "confidence_score": 1.0 - conf,
                    "hallucinated": False
                })
                
        return log


if __name__ == "__main__":
    # Test report generator
    from src.vision_encoder import GroundedVisionEncoder
    from src.grounding_module import GradCAMGrounding
    
    encoder = GroundedVisionEncoder()
    grounding = GradCAMGrounding(encoder)
    generator = ConstrainedReportGenerator(encoder, grounding)
    
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    out = generator.generate(dummy_img, patient_history="Shortness of breath")
    print(out.report_text)
    print(f"Evidence rows: {len(out.evidence_log)}")
