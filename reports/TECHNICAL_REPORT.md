# Technical Report: ClinicGuard ReportGen

ClinicGuard ReportGen is a research prototype for evidence-grounded chest X-ray report
generation. Its goal is to make generated text auditable: each claim should either be
supported by visual evidence, patient history, or a prior report, or the system should
refuse/hedge the claim.

This report describes the design and current runnable scope. It does not present the
prototype as clinically validated.

## 1. System Summary

Automated medical report generation is risky when a model can produce fluent but
unsupported clinical statements. ClinicGuard uses a constrained architecture:

- a DenseNet121-style vision encoder for pathology confidence scores,
- confidence thresholds for assertion, uncertainty, and refusal,
- Grad-CAM visual grounding for reported visual findings,
- template-based report construction,
- post-generation claim checks,
- CSV evidence logging for review.

The design favors traceability and controlled language over free-form generation.

## 2. Architecture

```text
Input X-ray
    |
    v
Vision encoder -> pathology confidence scores
    |
    v
Refusal / uncertainty gates
    |
    +--> Grad-CAM grounding for reportable findings
    |
    v
Template report builder
    |
    v
Claim verification + evidence log
    |
    v
Generated report, grounding overlays, CSV audit trail
```

## 3. Vision Encoder

The encoder uses a DenseNet121 backbone through `torchxrayvision` when available. The
project maps model outputs to 14 common chest X-ray pathology labels:

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Emphysema
- Fibrosis
- Hernia
- Infiltration
- Mass
- Nodule
- Pleural thickening
- Pneumonia
- Pneumothorax

If `torchxrayvision` is unavailable, the code falls back to a torchvision DenseNet so
the local pipeline remains inspectable, though that fallback should not be treated as a
trained medical model.

## 4. Refusal Gate

The report generator uses two thresholds:

- `evidence_threshold`: findings at or above this threshold can be asserted.
- `uncertainty_threshold`: findings between this and the assertion threshold are written
  with hedged language.

Findings below the uncertainty threshold are omitted from the positive report path. This
is the main control that prevents the generator from forcing a finding into the report
when the model confidence is low.

## 5. Grounding

For reportable visual findings, the grounding module creates a Grad-CAM heatmap and
extracts a bounding box from the strongest activation region. The evidence log stores
that source reference as an image-region string when available.

Grad-CAM is useful for auditability, but it is not a substitute for radiologist-labeled
segmentation or clinical validation.

## 6. Claim Verification

The hallucination detector checks generated sentences against available evidence:

- visual confidence scores,
- grounded regions,
- patient history text,
- prior report text.

The detector flags unsupported claims in the evidence log. This makes review easier, and
it also gives the evaluator a direct way to compute sample-level hallucination flags.

## 7. Current Data Support

The runnable public path focuses on IU X-Ray style data and bundled sample cases. MIMIC-CXR
and PadChest require external credentialing or approval, so the repository provides access
instructions and stubs rather than claiming automatic downloads.

This is intentional: protected medical datasets should not be represented as available
without the required approvals.

## 8. Evaluation Artifacts

The repository includes a small sample evidence log for demonstration:

- file: `reports/GROUNDING_EVIDENCE_LOG.csv`
- rows: 10 claim-level entries
- purpose: show the audit format and sample calculations

To produce a fresh benchmark, run:

```bash
python scripts/evaluate.py --num-samples 10 --output-dir evaluation/
```

A larger benchmark should include per-sample inputs, labels, predictions, generated text,
and claim-level evidence rows so the metrics can be independently inspected.

## 9. Known Limitations

- The project is not clinically validated.
- Demo fallback behavior can show simulated outputs when real model inference is not
  available.
- Protected datasets require manual setup and approvals.
- Template-based generation is safer than unconstrained prose, but less expressive.
- Grad-CAM boxes are approximate explanations, not ground-truth annotations.

## 10. Future Work

- Add reproducible per-sample benchmark exports.
- Add stricter claim parsing for negation and anatomical location.
- Add dataset adapters once protected datasets are available locally.
- Add calibrated uncertainty estimates before assertion thresholds are trusted.
- Add documentation for expected model checkpoints and output directories.
