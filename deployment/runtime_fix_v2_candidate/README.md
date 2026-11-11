# Runtime correction v2 — verified Pi video run

No training or model-weight changes. Keep both Python files together and use the existing RGB5 HEF with its deployment manifest.

Changes: the single-target display draws either an associated confirmed detection or the optical-flow box, never both. CONFIRMING/unmatched proposals are retained in experiment logs but are not presented as confirmed targets. A tightly contained detector box can reacquire the same oversized flow track rather than leaving the old large box active.

PC regression checks cover the reported screenshots, nested reacquisition, separated detections, real optical-flow motion and expiry. Saved actual Pi detector outputs were replayed through the new tracking/rendering on PC. That initial replay was followed by actual Pi verification on both models and both clips. status.json and results/rgb5_v2/runtime_fix_v2 record the separate scopes.

This correction does not turn raw building-edge false positives into correct detections or resolve weak drone subtype recognition. Single-target filtering is distinct from improving detector accuracy. The earlier v1 Pi videos remain historical evidence of the identified defect. The fresh Pi test is complete; it verifies this runtime on the recorded clips, not arbitrary-scene or live-camera reliability.

Verified update: runtime v2 has now been rerun on Raspberry Pi / Hailo-8 with both unchanged models and two 200-frame clips each. Small-target views were OFF. See results/rgb5_v2/runtime_fix_v2 for scope, raw logs, displayed-box scoring and remaining weak detection. The candidate folder name is retained for historical links; status.json now records Pi verification.
