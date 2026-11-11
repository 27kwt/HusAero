# Verified Raspberry Pi / Hailo-8 evaluation

500 actual images per model. Real Hailo-8, 640x640 RGB, compiled NMS confidence 0.001 / IoU 0.45, maximum 300 proposals/class; P/R at confidence 0.25. AP uses IoU 0.50:0.95, confidence ranking and 101 recall points. Separately labelled from native Ultralytics full-test metrics. Stratified sample is not population weighted. No test loss is calculated by the compiled HEF.

| Model | Split | Images | P @0.25 | R @0.25 | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HusAero-8n | test | 500 | 93.14% | 94.03% | 96.51% | 70.75% |
| HusAero-11n | test | 500 | 94.80% | 96.06% | 97.14% | 72.96% |

Training/validation results must not be represented as held-out test performance. Raw predictions and labels, hashes, split scores and actual curves are included. No retraining or test loss is claimed. Inference time measures synchronous Hailo inference including its postprocessing; it is not live-camera delay or video FPS. Hardware verification is for RGB image inference, not a new Pi tracking video.
