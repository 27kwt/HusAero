# Current RGB4 models

This folder documents the completed 100-epoch checkpoints from 14 November 2026: HusAero-8n (YOLOv8n) and HusAero-11n (YOLO11n). Both have four native outputs: `drone`, `bird`, `airplane`, `helicopter`. No subtype mapping adapter is needed for these PT files.

The model paths, byte counts and SHA-256 fingerprints are in [the registry](../registry.json). [Training and validation evidence](../../results/rgb4/) includes all 100 CSV rows per model, curves, recorded arguments and the preflight report. Compatible tensors were transferred from each earlier RGB5 checkpoint into a new four-class head; the two models have different prior training histories.

The [RGB4 PC Test evaluation](../../results/rgb4/test_20261117/), [ONNX/HEF compilation](../../results/rgb4/hailo8_compilation_20261114/) and [recorded-video Raspberry Pi 5 + Hailo-8 (26 TOPS) comparison](../../results/rgb4/pi_recorded_video_20261114/) are complete. HEF files remain in the local deployment bundle. Labelled HEF Test accuracy and live-camera/pan-tilt validation remain pending. The HEFs in `models/rgb5_v2` are older five-output models and cannot be substituted for these checkpoints.

## Download the checkpoints

- [HusAero-8n RGB4 best checkpoint](HusAero-8n_RGB4_best.pt)
- [HusAero-11n RGB4 best checkpoint](HusAero-11n_RGB4_best.pt)

Both PT files are included in this folder. Their bytes match the locally verified best checkpoints and the SHA-256 fingerprints in the registry. Run `python verify_release.py` from the repository root to verify the downloaded files before using the PC demo.
