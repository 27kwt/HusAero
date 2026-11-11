# Runtime v2 — actual Pi verification, no training

Both unchanged best HEFs processed the same two recorded RGB clips, 200 frames each, on drone-pi / Hailo-8. Host/device, source hashes, HEF hashes, output hashes and per-frame raw candidates plus displayed target boxes were verified. Four actual runs, 800 model-frames. Small-target views were OFF.

Correction: CONFIRMING/unassociated proposals are not shown as confirmed targets. Render either the associated detector result or the short-lived optical-flow box, never both. Contained fresh boxes can reacquire an oversized flow track. Only a single active target is displayed; all raw detector candidates remain in logs. Generic drone display does not correct the internal subtype classifier. No weights or image-benchmark metrics changed.

Offline annotation scoring uses original visible.json only after inference, IoU >=0.50 and confidence 0.25. First clip: follow localization 194/200 target-present frames for 8n, 136/200 for 11n. Difficult clip: 0/184 and 39/184. These are development clips, not an independent new holdout, general accuracy or a formal multi-object tracking benchmark. Training/session overlap is not excluded. Full numbers: annotated_v2.json. Detection and displayed-follow metrics are separate.

v1 optional-view tests are preserved as historical evidence, including the display defect. v2 fixes those display-state bugs but does not solve weak detection in the difficult clip. Corrected actual Pi comparison videos are shown beneath the unchanged older computer/Pi demonstration on the website.
