# RGB5 v2: completed PC image test

Fresh held-out evaluation of both best checkpoints from 50-epoch experiments on **all 2,159 test images**, 640-pixel input, RTX 4090. This is not Raspberry Pi/Hailo performance.

| Model | Precision | Recall | mAP50 | mAP50-95 | Mean drone AP50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HusAero-8n | 96.25% | 94.72% | 97.89% | 71.57% | 98.21% |
| HusAero-11n | 95.31% | 94.41% | 97.58% | 71.67% | 97.32% |

8n leads most of these metrics, including mean AP50 of the two drone classes; 11n leads mAP50-95 by about 0.10 percentage point. The differences do not establish an overall field-performance winner.

![Both models' actual test curves](Both_models_test_curves.png)

Each model folder contains test curves, confusion matrices, predictions, metric JSON and CSV curve values. Training_reference contains the actual original per-epoch training/validation graph and CSV, separately labelled. Test curves vary confidence thresholds; they are not per-epoch test losses. Reported mAP uses Ultralytics' native evaluation; the optional COCO evaluator has a different max-detections protocol. Precision/recall use the best mean-F1 operating point, not fixed confidence 0.25. Mean drone AP50 averages the two drone-class AP50 values, not merged-class AP.

Current RGB5 video check: **one unique unannotated RGB clip, 200 frames per model**, basic centre association. These operation counts are not tracking accuracy. Both candidates misclassified a visible multirotor as fixed-wing in a reviewed frame. Earlier 18-clip results elsewhere belong to the previous model release and are not extra RGB5 clips.

Both compiled Hailo-8 models were verified on Raspberry Pi with a 500-image stratified test-only sample; see [Pi_sample500](Pi_sample500/). This actual sample is separate from the full 2,159-image native PC test above. A complete-dataset Pi run reports training, validation and test separately; no unreturned result is claimed. Complete source recording-session independence is not established; prior checkpoint histories differ.

## Complete Raspberry Pi evaluation and videos

[Pi_all_images](Pi_all_images/) contains the verified evaluation of all 19,536 images per model (39,072 model-image evaluations): 13,954 training, 3,423 validation and 2,159 test, scored separately. On the held-out Pi test, 8n mAP50 / mAP50-95 = 95.42% / 68.48%, and 11n = 97.35% / 72.20%. 11n leads image-test quality on this hardware; 8n has lower mean synchronous inference time (8.73 ms versus 13.79 ms). These timings are not live-camera latency.

[Video evidence](Pi_all_images/video_evidence/) records two RGB clips on both models, 200 frames per run. The second clip was difficult: 8n produced detections in 1/200 frames and 11n in 11/200. These are unannotated operation counts, not measured recall. Both candidates still show subtype confusion in the other clip. The two actual side-by-side Pi comparisons are available beneath the existing video on the website Camera and Tracking page.

## Code-only runtime correction (no training)

[Runtime correction evidence](runtime_fix_20261111/) covers duplicate suppression, generic drone display, bounded LK tracking and the optional small-target experiment. Actual Pi runs showed zero duplicate-box frames. Optional views increased false positives for 8n, so they remain OFF by default. Difficult-video detection and subtype classification are not solved; no new benchmark score or trained weight is claimed. Original visible.json annotations were located and used offline for this regression comparison; the earlier video records had only reported unscored operation counts.

Verified update: runtime v2 has now been rerun on Raspberry Pi / Hailo-8 with both unchanged models and two 200-frame clips each. Small-target views were OFF. See results/rgb5_v2/runtime_fix_v2 for scope, raw logs, displayed-box scoring and remaining weak detection. The candidate folder name is retained for historical links; status.json now records Pi verification.
