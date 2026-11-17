# Current RGB4 Hailo-8 compilation evidence

Both current 100-epoch RGB4 checkpoints were exported to ONNX and compiled successfully for Hailo-8 (26 TOPS) on 14 November 2026. This folder publishes the recorded reports and checks the chain from the GitHub PT fingerprints to ONNX and HEF. The local HEF files were rehashed when preparing this evidence index.

[Fingerprint chain](fingerprints.json) · [Recorded-video Pi execution](../pi_recorded_video_20261114/)

The reports specify four output classes and RGB uint8 NHWC 640 × 640 input with letterbox padding 114. Calibration used 1,024 images. HEF binaries remain in the local deployment bundle and are not added by this results-only update. Successful compilation and the recorded-video run do not replace a labelled HEF Test evaluation or live-camera/pan-tilt validation.
