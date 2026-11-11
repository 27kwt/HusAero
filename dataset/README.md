# HusAero dataset versions

## Current: unified drone (RGB4)

Use [combined_rgb4](combined_rgb4/) for the current labels: **0 drone, 1 bird, 2 airplane, 3 helicopter**. Both fixed-wing and multirotor source objects map to drone. All 19,536 RGB images retain their original split: **13,954 train / 3,423 validation / 2,159 test**.

The remapped labels and provenance manifest are included in that folder. Image files are shared from [combined_rgb5_v2/images](combined_rgb5_v2/images/) so GitHub contains one copy of each canonical image. Run `python tools/prepare_current_dataset.py` from the repository root to create the local YOLO image/label layout. No training starts.

The current label merge is prepared, but the existing completed 50-epoch model weights remain native RGB5. Use the documented prediction adapter; no new four-output checkpoint or merged mAP score is claimed.

## Previous training data: native RGB5

[combined_rgb5_v2](combined_rgb5_v2/) preserves the exact old labels, images and evidence for the completed runs: 0 fixed_wing_drone, 1 multirotor_drone, 2 bird, 3 airplane, 4 helicopter. Original source licenses, grouping and file audit remain there. Keeping these records is essential for interpreting the historical metrics.

## Earlier starter subset

The root `images/`, `labels/` and `data.yaml` belong to the earlier **1,700-image** generic-drone/kite release (1,200 / 250 / 250), not the current combined dataset. Those IDs are 0 drone, 1 bird, 2 airplane, 3 helicopter, 4 kite. They stay at their old paths for old release code. Do not concatenate these labels with RGB4 or RGB5 labels. See [archived guide](README_RGB5_ARCHIVE.md).
