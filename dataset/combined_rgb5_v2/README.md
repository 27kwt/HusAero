# HusAero RGB5 v2

RGB / visible camera images only. No infrared or thermal data is used.

Class IDs: `0 fixed_wing_drone`, `1 multirotor_drone`, `2 bird`, `3 airplane`, `4 helicopter`.

The dataset has 19,536 images: 13,954 training, 3,423 validation and 2,159 test images. `summary.json` records the class counts and exclusions; `manifest.jsonl` records each selected image's source and partition.

Sources:

- [Yaren Koç fixed-wing UAV dataset](https://github.com/yarenk/fixed-wing-uav-dataset-and-model), dataset CC BY 4.0. One color variant is retained per original image family.
- [Nyahmet fixed-wing UAV dataset](https://www.kaggle.com/datasets/nyahmet/fixed-wing-uav-dataset), listed as CC0. Original-name aliases are grouped with the first fixed-wing source.
- [Simeon Ajakwe's VisioDECT](https://www.kaggle.com/datasets/simeonajakwe/visiodect-for-scenario-based-multi-drone-detection), CC BY-NC-SA 4.0. All six source aircraft models are multirotors. Consecutive frames are thinned before balancing.
- [MBAOD v1](https://data.mendeley.com/datasets/zfc8hndczp/1), Firas Mohammed Kadhum, Abbas Abdulazeez Abdulhameed and Jameelah Harbi Saud, CC BY 4.0. Only images whose annotations contain the retained bird, airplane and helicopter classes are used. Images containing generic UAV, kite or parachute annotations are excluded entirely.

Images are resized with padding to 640 × 640; boxes are transformed with them. Unreadable or unlabeled candidates, grayscale-only images, targets smaller than two pixels after resizing, repeated fixed-wing augmentation variants and conflicting exact duplicates are excluded. Class volumes are capped at approximately 4,000 images per primary class to avoid one source dominating the comparison. Augmentation during model training applies only to training data.

VisioDECT model/scenario groups stay in a single partition. Fixed-wing augmentation families and numbered neighborhoods stay together, with boundary frames purged. The original MBAOD holdout is preserved with protection for repeated original names. These measures prevent the recorded groups and exact duplicates crossing partitions; complete recording-session independence cannot be established from all public-source metadata.

Both models use the same files, partitions and training settings. Their released starting checkpoints have different prior histories, so the comparison does not isolate architecture alone. Test data is evaluated only after selecting each checkpoint using validation data. Public-source images are not footage recorded by this project team.

The source licenses remain applicable; this dataset includes noncommercial, share-alike material from VisioDECT. The original project datasets and checkpoints are preserved separately.

Independent RGB video check: both RGB5 candidates detected a visibly multirotor drone but labelled it fixed_wing_drone in a reviewed frame. Subtype recognition outside these sources needs further validation. These models are experimental.
