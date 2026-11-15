# Project dataset

This is the single dataset used for the current HusAero RGB4 models. Open `images` for photos or `labels` for their matching YOLO annotations. There is no separate RGB4/RGB5 image preparation step.

```text
dataset/
  images/       train/  val/  test/
  labels/       train/  val/  test/
  data.yaml     one training/evaluation definition
  provenance/   sources, licenses, counts and per-file checksums
```

| Split | Images | Labels |
| --- | ---: | ---: |
| Train | 13,954 | 13,954 |
| Validation | 3,423 | 3,423 |
| Test | 2,159 | 2,159 |
| **Total** | **19,536** | **19,536** |

Class IDs: **0 drone, 1 bird, 2 airplane, 3 helicopter**. Both drone subtypes are included in `drone`. The dataset contains visible-light RGB camera images.

An image such as `images/train/example.jpg` pairs with `labels/train/example.txt`. Each label line is `class_id x_center y_center width height`, with normalized coordinates. The folders are already ready for YOLO. To verify the downloaded files without training:

```sh
python tools/prepare_current_dataset.py
```

[Sources and preparation](provenance/DATA_CARD.md) · [Licenses and attribution](provenance/LICENSE_DATA.md) · [Audit](provenance/FILE_AUDIT.json) · [Counts](provenance/summary.json)

The reorganization changes paths only: all 19,536 image files, RGB4 labels and train/validation/test assignments are preserved. The obsolete starter dataset and separate RGB5 label directory are no longer in this folder. Original versions remain available through [the experiment archive](../archive/README.md). Use that frozen version to reproduce RGB5; its five-class labels must not be mixed with these four-class labels.
