# Dataset licenses and attribution

The project code license does not replace the image-source licenses.

| Source | Creator | Source license |
| --- | --- | --- |
| [Fixed-wing UAV dataset](https://github.com/yarenk/fixed-wing-uav-dataset-and-model) | Yaren Koç | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [Fixed Wing UAV Dataset](https://www.kaggle.com/datasets/nyahmet/fixed-wing-uav-dataset) | Nyahmet | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |
| [VisioDECT](https://www.kaggle.com/datasets/simeonajakwe/visiodect-for-scenario-based-multi-drone-detection) | Simeon Okechukwu Ajakwe and collaborators | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| [MBAOD v1](https://data.mendeley.com/datasets/zfc8hndczp/1) | Firas Mohammed Kadhum, Abbas Abdulazeez Abdulhameed and Jameelah Harbi Saud | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Changes: source labels were remapped to the five RGB5 classes; images and boxes were resized with padding to 640 × 640; candidates were filtered, grouped and balanced as described in README.md. Adapted VisioDECT material is distributed under CC BY-NC-SA 4.0. Preserve its attribution and noncommercial/share-alike terms. The other sources retain their stated licenses. `manifest.jsonl` identifies the source of each image and records final image/label checksums.

The public manifest removes machine-specific absolute paths and adds checksums for the copied files. `summary.json` preserves the original training-manifest fingerprint for the recorded experiments.
