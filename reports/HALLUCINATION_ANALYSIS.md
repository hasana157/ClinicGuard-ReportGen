# Hallucination Analysis

This note explains how ClinicGuard ReportGen defines, reduces, and audits unsupported
claims in generated radiology-style reports.

The project is a prototype. The included numbers are sample-artifact checks unless they
are regenerated from a documented evaluation run.

## 1. Why Hallucination Control Matters

In medical report generation, a hallucination is a clinically meaningful statement that is
not supported by the available evidence. In this project, evidence can come from:

- the image classifier confidence score,
- a Grad-CAM image region,
- structured patient history supplied by the user,
- a prior report supplied by the user.

The system is designed to make unsupported claims visible rather than hiding them inside
fluent prose.

## 2. Hallucination Types

Common failure modes include:

- Fabricated finding: reporting a pathology that is not supported by image evidence.
- Anatomical shift: assigning a finding to the wrong region or side.
- Negation inversion: turning "no pneumothorax" into "pneumothorax present."
- Severity distortion: overstating the size or severity of a finding.
- Source mismatch: using patient history or prior text as if it were current visual
  evidence.

## 3. Mitigation Strategy

ClinicGuard uses three controls:

1. Confidence-gated findings: low-confidence findings are omitted or hedged.
2. Template-based generation: report text is assembled from controlled clinical phrases.
3. Evidence logging: generated claims are linked back to a source reference and checked
   after generation.

This does not guarantee clinical correctness, but it does make unsupported output easier
to detect and review.

## 4. Sample Evidence Check

The bundled `reports/GROUNDING_EVIDENCE_LOG.csv` contains 10 sample claim rows. One row is
marked as hallucinated and uses an `UNGROUNDED` source reference. From this sample artifact:

| Sample Measure | Value |
| --- | ---: |
| Claim rows | 10 |
| Flagged hallucinated rows | 1 |
| Sample hallucination flag rate | 10% |
| Rows with non-ungrounded source references | 9 |
| Sample grounded-reference rate | 90% |

These values are useful for demonstrating the audit format. They are not a clinical
performance benchmark.

## 5. Recommended Evaluation Standard

A credible benchmark should save:

- sample IDs,
- source images or stable image references,
- ground-truth labels,
- model confidence scores,
- generated reports,
- evidence logs,
- hallucination flags,
- exact command and configuration used for the run.

Without those artifacts, precision, recall, grounding, and hallucination metrics should be
described as examples rather than final performance claims.

## 6. Practical Review Checklist

When reviewing generated output:

- Check every positive finding has either a grounded visual source or an explicitly named
  non-visual source.
- Check absent findings are not contradicted by high-confidence positive predictions.
- Check hedged findings are not presented as certain in the impression.
- Check prior-report language is not copied as a current image finding.
- Check each evidence row has a useful `source_reference`.
