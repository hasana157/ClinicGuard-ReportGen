"""
Clinically-validated radiology report templates.

Defines positive, uncertain, and negative templates for all 14 primary pathologies
and provides helper functions to assemble structured report findings and impression sections.
"""

from typing import Dict, List, Any, Optional

# ============================================================================
# Pathology Templates (Positive, Uncertain, Negative)
# ============================================================================

FINDING_TEMPLATES = {
    "Atelectasis": {
        "positive": "Subsegmental atelectasis is noted at the lung bases, causing mild volume loss.",
        "uncertain": "Borderline increased density at the lung bases may represent mild atelectasis. Recommend correlation with clinical history.",
        "negative": "No evidence of atelectasis is identified."
    },
    "Cardiomegaly": {
        "positive": "Cardiomegaly is identified with the cardiac silhouette measuring above normal limits.",
        "uncertain": "The cardiac silhouette appears borderline enlarged, which may represent mild cardiomegaly. Clinical correlation is recommended.",
        "negative": "The cardiac silhouette and mediastinal contours are within normal limits."
    },
    "Consolidation": {
        "positive": "Focal airspace consolidation is identified, suggestive of a localized pulmonary process.",
        "uncertain": "Patchy opacity is noted, which may represent developing consolidation. Close interval follow-up is suggested.",
        "negative": "No focal consolidation is seen."
    },
    "Edema": {
        "positive": "Prominence of the pulmonary vasculature and diffuse interstitial markings are consistent with pulmonary edema.",
        "uncertain": "Mild venous engorgement is noted, which may represent early or mild pulmonary edema. Clinical correlation for fluid status is suggested.",
        "negative": "No pulmonary edema is identified."
    },
    "Effusion": {
        "positive": "Pleural effusion is identified, associated with blunting of the costophrenic angle.",
        "uncertain": "Slight blunting of the costophrenic angle is noted, which could represent a small pleural effusion or pleural thickening.",
        "negative": "No pleural effusion is seen."
    },
    "Emphysema": {
        "positive": "Hyperinflation of the lungs with flattening of the diaphragms is consistent with emphysema.",
        "uncertain": "Mild hyperinflation is noted, which may represent early emphysematous changes.",
        "negative": "No emphysematous changes or hyperinflation are present."
    },
    "Fibrosis": {
        "positive": "Reticular opacities and volume loss are suggestive of pulmonary fibrosis.",
        "uncertain": "Fine reticular markings are noted, which may represent early interstitial fibrosis or scarring. Clinical correlation is advised.",
        "negative": "No pulmonary fibrosis or significant scarring is identified."
    },
    "Hernia": {
        "positive": "A retrocardiac soft tissue mass is consistent with a hiatal hernia.",
        "uncertain": "A small retrocardiac density is noted, possibly representing a small hiatal hernia.",
        "negative": "No hiatal hernia is seen."
    },
    "Infiltration": {
        "positive": "Patchy infiltrates are identified in the lung fields, consistent with airspace disease.",
        "uncertain": "Vague interstitial markings are noted, which may represent early infiltration or resolving infection.",
        "negative": "No focal pulmonary infiltrates are identified."
    },
    "Mass": {
        "positive": "A well-defined soft tissue mass is identified in the lung field, measuring above nodular threshold. Further evaluation with CT is recommended.",
        "uncertain": "A vague soft tissue density is noted, which may represent a superimposed shadow or a mass. Recommend CT correlation.",
        "negative": "No mass lesions are identified."
    },
    "Nodule": {
        "positive": "A small nodular opacity is noted in the lung parenchyma. Follow-up is recommended based on Fleischner Society guidelines.",
        "uncertain": "A tiny focal density is seen, which may represent a sub-centimeter nodule or vessel on-end.",
        "negative": "No pulmonary nodules are identified."
    },
    "Pleural_Thickening": {
        "positive": "Pleural thickening is identified, localized along the pleural margin.",
        "uncertain": "Borderline pleural thickening is noted, possibly representing old scarring.",
        "negative": "No pleural thickening is seen."
    },
    "Pneumonia": {
        "positive": "Focal consolidation and airspace opacity are highly suggestive of pneumonia. Clinical correlation with inflammatory markers is recommended.",
        "uncertain": "Patchy opacity is noted, which may represent an early or atypical pneumonia. Recommend clinical correlation.",
        "negative": "No findings suggestive of pneumonia are identified."
    },
    "Pneumothorax": {
        "positive": "A pneumothorax is identified with a visible pleural line and absence of lung markings. This requires urgent clinical correlation.",
        "uncertain": "A small apical pleural line is suspected, which may represent a tiny pneumothorax. Expiratory film or close observation is suggested.",
        "negative": "No pneumothorax is identified."
    }
}


# ============================================================================
# Section Layout & Headers
# ============================================================================

SECTION_TEMPLATES = {
    "INDICATION": "INDICATION: Chest pain and shortness of breath.",
    "COMPARISON": "COMPARISON: None available.",
    "FINDINGS_HEADER": "FINDINGS:",
    "IMPRESSION_HEADER": "IMPRESSION:"
}

NORMAL_STUDY_TEMPLATE = """FINDINGS:
The cardiothoracic ratio is within normal limits. The mediastinal contours are normal. The lungs are clear without focal consolidation, effusion, or pneumothorax. There is no evidence of active cardiopulmonary disease. The osseous structures are intact.

IMPRESSION:
No acute cardiopulmonary abnormality."""


# ============================================================================
# Helper Functions
# ============================================================================

def get_finding_template(
    finding: str, 
    confidence: float, 
    threshold_positive: float = 0.75, 
    threshold_uncertain: float = 0.50
) -> str:
    """
    Get the appropriate template string based on classification confidence.

    Args:
        finding: The name of the pathology.
        confidence: Prediction confidence score (0 to 1).
        threshold_positive: Confidence threshold to assert the finding is present.
        threshold_uncertain: Confidence threshold to hedge the finding.

    Returns:
        The template string.
    """
    if finding not in FINDING_TEMPLATES:
        return ""
        
    templates = FINDING_TEMPLATES[finding]
    if confidence >= threshold_positive:
        return templates["positive"]
    elif confidence >= threshold_uncertain:
        return templates["uncertain"]
    else:
        return templates["negative"]


def build_findings_section(
    predictions: Dict[str, float],
    threshold_positive: float = 0.75,
    threshold_uncertain: float = 0.50
) -> str:
    """
    Assemble the FINDINGS section based on predicted findings.

    Args:
        predictions: Dict of finding_name -> confidence.
        threshold_positive: Positive threshold.
        threshold_uncertain: Uncertain threshold.

    Returns:
        Assembled findings text.
    """
    # Find active positive and uncertain findings
    positive_findings = [f for f, c in predictions.items() if c >= threshold_positive]
    uncertain_findings = [f for f, c in predictions.items() if threshold_uncertain <= c < threshold_positive]
    
    # If it's a completely normal study
    if not positive_findings and not uncertain_findings:
        return NORMAL_STUDY_TEMPLATE.split("\n\n")[0].replace("FINDINGS:\n", "")
        
    # Standard reports list sentences
    findings_sentences = []
    
    # Process positive findings first
    for f in positive_findings:
        findings_sentences.append(FINDING_TEMPLATES[f]["positive"])
        
    # Process uncertain findings
    for f in uncertain_findings:
        findings_sentences.append(FINDING_TEMPLATES[f]["uncertain"])
        
    # Add boilerplate negatives for high-stakes findings if they aren't positive/uncertain
    critical_negatives = ["Pneumonia", "Pneumothorax", "Effusion"]
    for f in critical_negatives:
        if f not in positive_findings and f not in uncertain_findings:
            findings_sentences.append(FINDING_TEMPLATES[f]["negative"])
            
    # Combine
    return " ".join(findings_sentences)


def build_impression_section(
    predictions: Dict[str, float],
    threshold_positive: float = 0.75,
    threshold_uncertain: float = 0.50
) -> str:
    """
    Assemble the IMPRESSION section based on predicted findings.

    Args:
        predictions: Dict of finding_name -> confidence.
        threshold_positive: Positive threshold.
        threshold_uncertain: Uncertain threshold.

    Returns:
        Assembled impression text.
    """
    positive_findings = [f for f, c in predictions.items() if c >= threshold_positive]
    uncertain_findings = [f for f, c in predictions.items() if threshold_uncertain <= c < threshold_positive]
    
    if not positive_findings and not uncertain_findings:
        return NORMAL_STUDY_TEMPLATE.split("\n\n")[1].replace("IMPRESSION:\n", "")
        
    impression_items = []
    
    # Positive impression items
    for i, f in enumerate(positive_findings):
        # Human-readable formatting
        name = f.replace("_", " ")
        impression_items.append(f"{i+1}. {name}.")
        
    # Uncertain impression items
    offset = len(positive_findings)
    for i, f in enumerate(uncertain_findings):
        name = f.replace("_", " ").lower()
        impression_items.append(f"{offset + i + 1}. Borderline or possible {name} — recommend clinical correlation.")
        
    return "\n".join(impression_items)


def build_full_report(
    predictions: Dict[str, float],
    patient_history: Optional[str] = None,
    prior_report: Optional[str] = None,
    threshold_positive: float = 0.75,
    threshold_uncertain: float = 0.50
) -> str:
    """
    Build a complete, structured radiology report.

    Args:
        predictions: Dict of finding_name -> confidence.
        patient_history: Patient clinical details.
        prior_report: Previous report description.
        threshold_positive: Positive threshold.
        threshold_uncertain: Uncertain threshold.

    Returns:
        Complete formatted report text.
    """
    indication = f"INDICATION: {patient_history}" if patient_history else SECTION_TEMPLATES["INDICATION"]
    comparison = f"COMPARISON: {prior_report}" if prior_report else SECTION_TEMPLATES["COMPARISON"]
    
    findings = build_findings_section(predictions, threshold_positive, threshold_uncertain)
    impression = build_impression_section(predictions, threshold_positive, threshold_uncertain)
    
    report = f"{indication}\n{comparison}\n\nFINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
    return report


if __name__ == "__main__":
    # Test templates
    mock_preds = {
        "Cardiomegaly": 0.85,
        "Effusion": 0.60,
        "Pneumothorax": 0.10
    }
    
    print(build_full_report(mock_preds))
