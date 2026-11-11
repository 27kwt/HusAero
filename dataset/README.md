# HusAero datasets

## Current experiment: RGB5 v2 (50 epochs per model)

The complete combined RGB-only dataset is in [combined_rgb5_v2](combined_rgb5_v2/). Images and matching YOLO labels are included directly in this repository.

| Split | Images | Labels |
| --- | ---: | ---: |
| Training | 13,954 | 13,954 |
| Validation | 3,423 | 3,423 |
| Test | 2,159 | 2,159 |
| Total | 19,536 | 19,536 |

Class IDs: **0 fixed_wing_drone, 1 multirotor_drone, 2 bird, 3 airplane, 4 helicopter**. Images are 640 × 640 RGB. Kite is excluded. Class counts are not equal; images may contain multiple classes. See [summary.json](combined_rgb5_v2/summary.json).

Use **[combined_rgb5_v2/data.yaml](combined_rgb5_v2/data.yaml)** with the RGB5 model checkpoints. Each image in `combined_rgb5_v2/images/<split>/` has a matching label in `combined_rgb5_v2/labels/<split>/`.

### Sources and combination

Four source datasets form three task groups: two fixed-wing sources (Yaren Koç and Nyahmet), VisioDECT for multirotors, and MBAOD v1 for birds, airplanes and helicopters. Labels were remapped into one five-class map, images and boxes letterboxed together, unsuitable/removed-class images and duplicates excluded, and the retained files grouped and split. The resulting files are already combined; there is no need to concatenate the old folders.

DUT Anti-UAV was used in earlier checkpoint training; it is **not an image source of this RGB5 v2 dataset**. Fine-tuning starts from released HusAero checkpoints with earlier training histories, not from scratch.

See the [preparation details](combined_rgb5_v2/README.md), [data card](combined_rgb5_v2/DATA_CARD.md), [source licenses](combined_rgb5_v2/LICENSE_DATA.md), [per-image provenance](combined_rgb5_v2/manifest.jsonl), and [full file audit](combined_rgb5_v2/FILE_AUDIT.json).

### Testing

Both RGB5 checkpoints were tested on **all 2,159 test images**, not a sample. These images were not used for checkpoint selection in the new fine-tuning runs. Actual PC curves and scores are in [results/rgb5_v2](../results/rgb5_v2/). A separately labelled stratified 500-image test was completed on the actual Raspberry Pi / Hailo-8 for both checkpoints; see [verified hardware evidence](../results/rgb5_v2/Pi_sample500/). Full-dataset hardware evaluation is a separate run, with Train, Validation and Test reported separately.

## Earlier release: MBAOD starter subset

The root `images/`, `labels/`, `data.yaml` and `curation_report.json` belong to the earlier generic-drone release: **1,200 train / 250 validation / 250 test**. Its IDs are **0 drone, 1 bird, 2 airplane, 3 helicopter, 4 kite**.

These folders stay in their original locations so the earlier released software and historical results remain reproducible. **Do not merge their labels into RGB5:** identical IDs mean different classes. They are not extra RGB5 samples. The root `data.yaml` is intentionally the legacy configuration.

The starter subset is from MBAOD v1 by Firas Mohammed Kadhum, Abbas Abdulazeez Abdulhameed and Jameelah Harbi Saud, publisher-listed CC BY 4.0, DOI [10.17632/zfc8hndczp.1](https://doi.org/10.17632/zfc8hndczp.1). Earlier released checkpoints also used DUT/video data not redistributed in this starter subset.

HusAero integration, data preparation, fine-tuning and evaluation: **Hussain Alrashidi (27kwt)**. Source creators retain attribution to their original images, annotations and software; see [NOTICE.md](../NOTICE.md).
