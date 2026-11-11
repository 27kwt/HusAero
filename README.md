# HusAero

RGB drone detection and single-target tracking on PC and Raspberry Pi / Hailo-8.
Project integration, dataset preparation, fine-tuning and evaluation: **Hussain Alrashidi (27kwt)**. Built on Ultralytics YOLOv8n and YOLO11n; upstream and image-source credits are in [NOTICE.md](NOTICE.md).

## Current focus: one drone category

Fixed-wing and multirotor drones are grouped as **drone**. The current dataset categories are **drone, bird, airplane and helicopter**. Bird/airplane/helicopter examples help distinguish other airborne objects; no kite class is included in this version.

The project detects a visual drone category. It does not identify a particular aircraft model, military/civilian status, payload or mission from appearance. Confusing scenes remain a model limitation. The drone-focused output adapter uses **unknown_airborne_object** for unverified non-drone types or low/conflicting scores, rather than asserting an aircraft type. These are conservative display rules, not calibrated certainty or a guarantee against misclassification.

## Dataset and preparation

| Split | RGB images | Matching labels |
| --- | ---: | ---: |
| Train | 13,954 | 13,954 |
| Validation | 3,423 | 3,423 |
| Test | 2,159 | 2,159 |
| Total | 19,536 | 19,536 |

Start with [the unified dataset](dataset/combined_rgb4/). All source images and the current labels are in this repository. Image bytes are shared with the [canonical image folder](dataset/combined_rgb5_v2/images/) instead of uploading duplicates. Sources, filtering, source groups, split limits and licenses are documented alongside the data.

```sh
python tools/prepare_current_dataset.py
```

The command verifies the file fingerprints and prepares YOLO's image/label layout. It does not start training. Current IDs: **0 drone, 1 bird, 2 airplane, 3 helicopter**.

## Models: what is completed

Both **HusAero-8n** and **HusAero-11n** completed the recorded 50-epoch RGB5 experiments. Their best PT checkpoints and Hailo-8 HEFs are in [models/rgb5_v2](models/rgb5_v2/), with hashes and verified runtime information.

Those existing weights still have **five native outputs**. The unified four-category labels are a later preparation change; **no four-output retraining has been run**. The adapter maps native IDs 0/1 to drone and shifts the remaining categories. It preserves native predictions in its log. Renaming a dataset does not change model weights or fix missed drones.

Use Python 3.11. For the tested CPU environment setup, see [the installation guide](README_RGB5_ARCHIVE.md#2-install-and-run-detection-on-a-pc).

Example using the existing 8n checkpoint (replace the source with your RGB photo or video):

```sh
python -m pip install -r requirements.txt
python tools/predict_unified.py --model models/rgb5_v2/HusAero-8n/HusAero-8n_RGB5.pt --source path/to/input.jpg --output results/unified_predictions.jsonl
```

Use HusAero-11n/HusAero-11n_RGB5.pt for the second model. The adapter is a PC output mapping and abstention layer. The verified Pi runtime remains the documented native RGB5 HEF pipeline, displaying generic drone and retaining native class IDs in experiment logs. A new four-output HEF has not been compiled.

## Results and actual operation evidence

- [Completed native RGB5 image tests and curves](results/rgb5_v2/): all 2,159 test images per model on PC; all 19,536 images per model on Pi with train/validation/test reported separately.
- [Verified Pi runtime v2](results/rgb5_v2/runtime_fix_v2/): two recorded RGB clips per model, 200 frames each, with per-frame detector proposals and the displayed target box.
- [Project website](https://fyp1-tracking-lab-hussain.lustrousdulse1.chatgpt.site/#camera): the earlier PC/Pi demonstration remains; the current two Pi model comparisons appear below it.

The existing mAP tables concern the native five-category models. **They are not new four-category scores.** The current relabeling/adapter has no newly measured training or test mAP. The difficult video still has weak detection, and source-session/training overlap is not ruled out. Recorded-file/image timings do not establish live-camera delay. Flow-assisted following is separated from fresh detection and is not advertised as general accuracy or formal MOT performance.

## Provenance and earlier versions

The RGB4 and RGB5 preparations use four public source datasets: Yaren Koç fixed-wing, Ahmet Nuri Yılmaz (nyahmet) fixed-wing, VisioDECT, and MBAOD v1. Full creators, source links, licenses and modifications: [dataset license notice](dataset/combined_rgb4/LICENSE_DATA.md). VisioDECT's recorded distribution includes noncommercial/share-alike terms; the code license does not replace image licenses. Public source imagery is not presented as our own photography.

Earlier generic-drone/kite release files and the original RGB5 labels remain for reproducibility; their class IDs must not be mixed with current IDs. [Archived project guide](README_RGB5_ARCHIVE.md), [dataset version guide](dataset/README.md), [model registry](models/rgb5_v2/registry.json). The old release_manifest.json describes its historical source snapshot, not all current additions.
