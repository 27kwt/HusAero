# Experimental HusAero RGB5 checkpoints

HusAero-8n (YOLOv8n) and HusAero-11n (YOLO11n), each fine-tuned for 50 epochs on the same RGB5 v2 dataset. Author of HusAero integration and fine-tuning: Hussain Alrashidi (27kwt). Ultralytics retains upstream architecture/software attribution; image licenses remain separate.

Class map: 0 fixed_wing_drone, 1 multirotor_drone, 2 bird, 3 airplane, 4 helicopter. These weights are incompatible with the old generic-drone class map. The earlier default remains unchanged.

Each subfolder contains its PT checkpoint, compiled Hailo-8 HEF, hash-checked deployment manifest, compilation metadata and RGB-video demo. Compiler output targets Hailo-8, not 8L or 10H. Both exact HEFs were subsequently verified on the actual Raspberry Pi / Hailo-8 using 500 labelled test images each. Read deployment.json, its pi_verification scope and the raw hardware evidence under results/rgb5_v2/Pi_sample500. Image inference verification is not a Pi video-tracking benchmark.

PC image test from the repository root:

```sh
python -m pip install ultralytics==8.4.201
yolo val model=models/rgb5_v2/HusAero-8n/HusAero-8n_RGB5.pt data=dataset/combined_rgb5_v2/data.yaml split=test imgsz=640
yolo val model=models/rgb5_v2/HusAero-11n/HusAero-11n_RGB5.pt data=dataset/combined_rgb5_v2/data.yaml split=test imgsz=640
```

Select a local CUDA device when available, or add device=cpu. Original training arguments and per-epoch metrics are under results/rgb5_v2/<model>/Training_reference. Reproducing the exact fine-tuning also requires the earlier starting checkpoint and class-head remapping; PT files here are the completed outputs, not fresh baseline weights.

Both candidates confused a multirotor with fixed-wing in an independent RGB video. Use as experimental models; the image benchmark is not a guarantee of reliable subtype recognition.

## Runtime correction v1

The trained best PT/HEF bytes are unchanged. Keep flow_assisted_tracker.py beside husaero_rgb5_live.py. The display now uses generic drone because subtype recognition is not verified. Raw predicted classes remain in the experiment logs. Overlapping/nested drone boxes are suppressed before association. Short LK-flow boxes are orange and explicitly labelled as having no new detector result. Optional --small-target views remain disabled by default because the recorded Pi diagnostic increased false positives for 8n. See results/rgb5_v2/runtime_fix_20261111 for actual test scope and remaining limitations.

Verified update: runtime v2 has now been rerun on Raspberry Pi / Hailo-8 with both unchanged models and two 200-frame clips each. Small-target views were OFF. See results/rgb5_v2/runtime_fix_v2 for scope, raw logs, displayed-box scoring and remaining weak detection. The candidate folder name is retained for historical links; status.json now records Pi verification.
