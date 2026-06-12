"""
Hallucination detection module.

Extracts clinical claims (sentences) from generated reports and cross-references
each claim against visual evidence, patient history, and prior reports to identify
unsupported or fabricated assertions.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from src.config import HallucinationDetectorConfig, PATHOLOGY_LABELS, get_config


@dataclass
class HallucinationResult:
    """Dataclass storing the factuality check result for a single claim."""
    claim: str
    hallucinated: bool
    source_type: str  # visual, history, prior, or UNGROUNDED
    confidence: float
    explanation: str


class HallucinationDetector:
    """Post-generation factuality verification engine."""

    def __init__(self, config: Optional[HallucinationDetectorConfig] = None):
        """
        Initialize the detector.

        Args:
            config: Detector configuration.
        """
        self.config = config if config else HallucinationDetectorConfig()
        
        # Load synonym dictionary for claim matching
        from src.data_loader import PATHOLOGY_SYNONYMS
        self.synonyms = PATHOLOGY_SYNONYMS

    def extract_claims(self, report_text: str) -> List[str]:
        """
        Extract sentences/claims from report text, filtering out headers.

        Args:
            report_text: Full report string.

        Returns:
            List of claim sentences.
        """
        # Split into sentences using punctuation rules
        sentences = re.split(r"(?<=[.!?])\s+", report_text)
        
        claims = []
        for s in sentences:
            s_clean = s.strip()
            # Ignore headers and empty lines
            if not s_clean:
                continue
            if any(s_clean.upper().startswith(h) for h in ["FINDINGS:", "IMPRESSION:", "INDICATION:", "COMPARISON:"]):
                continue
            # Filter out numeric list markers in Impression (e.g. "1. Cardiomegaly")
            s_clean = re.sub(r"^\d+\.\s*", "", s_clean)
            if s_clean:
                claims.append(s_clean)
                
        return claims

    def verify_claim(self, claim: str, evidence: Dict[str, Any]) -> HallucinationResult:
        """
        Cross-reference a claim against all available evidence sources.

        Args:
            claim: Claim text.
            evidence: Dict containing 'visual' (predictions), 'history' (text), 'prior' (text).

        Returns:
            HallucinationResult.
        """
        claim_lower = claim.lower()
        
        # Check if the claim mentions "normal" or "no abnormalities"
        is_normal_assertion = any(w in claim_lower for w in ["normal", "clear", "no evidence", "without", "no pleural", "no pneumothorax", "no focal", "no mass"])
        
        # Match finding keywords
        matched_finding = None
        for finding, syns in self.synonyms.items():
            if any(syn in claim_lower for syn in syns):
                matched_finding = finding
                break
                
        # If no specific pathology is mentioned, assess global status
        if not matched_finding:
            if is_normal_assertion:
                # Normal claims are grounded by absence of positive findings
                visual_probs = evidence.get("visual", {})
                max_visual = max(visual_probs.values()) if visual_probs else 0.0
                if max_visual < 0.75:
                    return HallucinationResult(
                        claim=claim,
                        hallucinated=False,
                        source_type="visual",
                        confidence=1.0 - max_visual,
                        explanation="Grounded: Normal statement supported by absence of high-confidence pathologies."
                    )
            return HallucinationResult(
                claim=claim,
                hallucinated=False,
                source_type="visual",
                confidence=0.9,
                explanation="Grounded: General clinical description or boilerplate."
            )
            
        # 1. Verify against visual evidence
        visual_probs = evidence.get("visual", {})
        finding_prob = visual_probs.get(matched_finding, 0.0)
        
        # 2. Verify against patient history
        history_text = evidence.get("history", "")
        history_match = False
        if history_text:
            history_lower = history_text.lower()
            syns = self.synonyms.get(matched_finding, [matched_finding.lower()])
            history_match = any(syn in history_lower for syn in syns)
            
        # 3. Verify against prior reports
        prior_text = evidence.get("prior", "")
        prior_match = False
        if prior_text:
            prior_lower = prior_text.lower()
            syns = self.synonyms.get(matched_finding, [matched_finding.lower()])
            prior_match = any(syn in prior_lower for syn in syns)

        # Logic to decide if claim is hallucinated:
        # If the claim asserts presence of finding, check if prediction confidence is high enough
        # OR if history/prior report confirms it.
        # If the claim asserts absence or uncertainty, check appropriate thresholds.
        
        is_asserting_presence = not any(neg in claim_lower for neg in ["no", "without", "normal", "clear", "not seen", "absent", "negative"])
        
        if is_asserting_presence:
            # Requires visual confidence >= 0.5 (or history/prior match)
            if finding_prob >= 0.5 or history_match or prior_match:
                source = "visual" if finding_prob >= 0.5 else "history" if history_match else "prior"
                conf = finding_prob if source == "visual" else 1.0
                return HallucinationResult(
                    claim=claim,
                    hallucinated=False,
                    source_type=source,
                    confidence=conf,
                    explanation=f"Grounded: Presence of {matched_finding} verified by {source} evidence."
                )
            else:
                # Under-threshold assertion is considered a hallucination
                return HallucinationResult(
                    claim=claim,
                    hallucinated=True,
                    source_type="UNGROUNDED",
                    confidence=finding_prob,
                    explanation=f"Hallucination: Claim asserts {matched_finding} but visual confidence ({finding_prob:.1%}) is insufficient and no history/prior match exists."
                )
        else:
            # Asserting absence or mild/uncertain presence
            # Gronded if visual confidence is low (or history/prior confirms it)
            if finding_prob < 0.75 or history_match or prior_match:
                source = "visual" if finding_prob < 0.75 else "history" if history_match else "prior"
                conf = 1.0 - finding_prob if source == "visual" else 1.0
                return HallucinationResult(
                    claim=claim,
                    hallucinated=False,
                    source_type=source,
                    confidence=conf,
                    explanation=f"Grounded: Absence/uncertainty of {matched_finding} verified by {source} evidence."
                )
            else:
                return HallucinationResult(
                    claim=claim,
                    hallucinated=True,
                    source_type="UNGROUNDED",
                    confidence=finding_prob,
                    explanation=f"Hallucination: Claim asserts normal or absent {matched_finding} but visual confidence indicates high probability ({finding_prob:.1%})."
                )

    def detect_hallucinations(self, report_text: str, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verify all claims in a report and return structured dict list of results.

        Args:
            report_text: Full report text.
            evidence: Dict containing evidence sources.

        Returns:
            List of dicts containing check results.
        """
        claims = self.extract_claims(report_text)
        results = []
        for claim in claims:
            res = self.verify_claim(claim, evidence)
            results.append({
                "claim": res.claim,
                "hallucinated": res.hallucinated,
                "source_type": res.source_type,
                "confidence_score": float(res.confidence),
                "explanation": res.explanation
            })
        return results

    def compute_hallucination_rate(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate the percentage of claims flagged as hallucinations.

        Args:
            results: List of verification results.

        Returns:
            Hallucination rate float (0 to 1).
        """
        if not results:
            return 0.0
        hallucinated = sum(1 for r in results if r["hallucinated"])
        return hallucinated / len(results)


if __name__ == "__main__":
    # Test detector
    detector = HallucinationDetector()
    report = "FINDINGS:\nCardiomegaly is identified. Lungs are clear. No pneumothorax.\n\nIMPRESSION:\n1. Cardiomegaly."
    evidence = {
        "visual": {"Cardiomegaly": 0.85, "Effusion": 0.1, "Pneumothorax": 0.05},
        "history": "History of enlarged heart",
        "prior": ""
    }
    
    flags = detector.detect_hallucinations(report, evidence)
    for f in flags:
        print(f"Claim: {f['claim']}\nHallucinated: {f['hallucinated']}\nExplanation: {f['explanation']}\n")
    print(f"Hallucination Rate: {detector.compute_hallucination_rate(flags):.1%}")
