# Verified Raspberry Pi / Hailo-8 evaluation

19536 actual images per model. Real Hailo-8, 640x640 RGB, compiled NMS confidence 0.001 / IoU 0.45, maximum 300 proposals/class; P/R at confidence 0.25. AP uses IoU 0.50:0.95, confidence ranking and 101 recall points. Separately labelled from native Ultralytics full-test metrics. All dataset partitions evaluated and reported separately; only Test is held-out. No test loss is calculated by the compiled HEF.

| Model | Split | Images | P @0.25 | R @0.25 | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HusAero-8n | test | 2159 | 93.14% | 93.61% | 95.42% | 68.48% |
| HusAero-8n | train | 13954 | 93.16% | 95.51% | 96.65% | 69.32% |
| HusAero-8n | val | 3423 | 90.20% | 87.61% | 91.27% | 61.44% |
| HusAero-11n | test | 2159 | 93.93% | 94.69% | 97.35% | 72.20% |
| HusAero-11n | train | 13954 | 95.58% | 96.03% | 97.86% | 74.79% |
| HusAero-11n | val | 3423 | 90.80% | 87.71% | 92.68% | 65.92% |

Training/validation results must not be represented as held-out test performance. Raw predictions and labels, hashes, split scores and actual curves are included. No retraining or test loss is claimed. Inference time measures synchronous Hailo inference including its postprocessing; it is not live-camera delay or video FPS. Hardware verification is for RGB image inference, not a new Pi tracking video.
