# Runtime correction v2 — candidate, awaiting Pi verification

No training or model-weight changes. Keep both Python files together and use the existing RGB5 HEF with its deployment manifest.

Changes: the single-target display draws either an associated confirmed detection or the optical-flow box, never both. CONFIRMING/unmatched proposals are retained in experiment logs but are not presented as confirmed targets. A tightly contained detector box can reacquire the same oversized flow track rather than leaving the old large box active.

PC regression checks cover the reported screenshots, nested reacquisition, separated detections, real optical-flow motion and expiry. Saved actual Pi detector outputs were replayed through the new tracking/rendering on PC. This replay is not a new Pi inference or runtime verification. status.json records the verification boundary.

This correction does not turn raw building-edge false positives into correct detections or resolve weak drone subtype recognition. Single-target filtering is distinct from improving detector accuracy. The earlier v1 Pi videos remain historical evidence of the identified defect. A fresh Pi test is pending the user's local SSH login; do not label this candidate as Pi-verified until that test completes.
