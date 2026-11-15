# Current RGB4 models

This folder documents the completed 100-epoch checkpoints from 14 November 2026: HusAero-8n (YOLOv8n) and HusAero-11n (YOLO11n). Both have four native outputs: `drone`, `bird`, `airplane`, `helicopter`. No subtype mapping adapter is needed for these PT files.

The model paths, byte counts and SHA-256 fingerprints are in [the registry](../registry.json). [Training and validation evidence](../../results/rgb4/) includes all 100 CSV rows per model, curves, recorded arguments and the preflight report. Compatible tensors were transferred from each earlier RGB5 checkpoint into a new four-class head; the two models have different prior training histories.

Held-out RGB4 Test evaluation, RGB4 ONNX/HEF compilation, and Raspberry Pi 5 + Hailo-8 (26 TOPS) verification are still pending. The HEFs in `models/rgb5_v2` are older five-output models and cannot be substituted for these checkpoints.

## Checkpoint availability

The two PT files are verified locally but their GitHub upload is still pending. The large binary upload exceeded the automatic approval review capacity. This folder currently provides their documentation; the registry records their expected paths and SHA-256 fingerprints. Place the verified PT files here before using the RGB4 PC demo. Existing RGB5 weights remain available in `models/rgb5_v2`.
