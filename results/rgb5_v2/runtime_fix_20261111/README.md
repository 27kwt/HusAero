# Runtime correction v1 — no training

Exact best PT/HEF hashes remain unchanged. Corrections: drone-class duplicate suppression (including nested boxes); generic drone display with original predicted subtype retained in logs; fixed square PC preprocessing to match export; bounded Lucas-Kanade flow with orange FLOW boxes distinguished from new detections. Velocity-only propagation is not rendered as a measured target.

The four-view small-target fallback is optional and OFF by default. Actual testing of that option on both models and the two original 200-frame clips eliminated duplicate boxes, but increased false positives for 8n. It is not a general improvement and is not recommended as the default. Difficult-video detection and subtype recognition remain unresolved model limitations.

The before/after table is annotated_comparison.csv. Scoring uses the original single-target visible.json AFTER inference only, confidence 0.25, IoU >=0.50, and at most one target match per frame. Subtypes merge into generic drone for localization scoring only. The clips are development/regression inputs and are not a new independent holdout. Other visible objects may be unannotated. Flow counts are not tracking accuracy. No image-benchmark score or test curve was recomputed by this runtime test.

At 8n on the first clip, the diagnostic reduced duplicate frames from 14 to 0 but TP stayed 127 and FP increased from 14 to 25 with optional views. At 11n, duplicate frames fell from 5 to 0, TP rose from 84 to 142, and FP stayed 27. The difficult clip remained weak: TP 1/184 target-present frames for 8n and 16/184 for 11n. See the full table rather than interpreting visible tracking as detector recall.

Run a model from its own model folder with the matching HEF and deployment.json, using husaero_rgb5_live.py --hef <model>.hef --manifest deployment.json --video <RGB input> --port 8084. flow_assisted_tracker.py must stay beside it. Add --small-target only for a separately evaluated experimental run, accepting extra latency and false positives. This is a code update, not a new trained model.

## Subsequent display defect and pending v2

The detector-level NMS checks did not catch simultaneous drawing of a FLOW box and an unmatched detector proposal. A CONFIRMING building-edge proposal was also incorrectly drawn in frame 72. The v1 videos retain these issues as historical evidence. Candidate v2 under deployment/runtime_fix_v2_candidate fixes mutually exclusive single-target display, hides temporally unconfirmed proposals from the confirmed-target view, and supports contained-box reacquisition. PC tests and replay passed; v2 has not yet been rerun on the Pi. Underlying raw detector false positives remain in logs, and no retraining took place.
