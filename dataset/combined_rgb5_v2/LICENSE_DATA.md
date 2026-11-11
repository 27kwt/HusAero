# Dataset licenses and attribution

The project code license does not replace the image-source licenses.

| Source | Creator | Source license |
| --- | --- | --- |
| [Fixed-wing UAV dataset](https://github.com/yarenk/fixed-wing-uav-dataset-and-model) | Yaren Koç | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [Fixed Wing UAV Dataset](https://www.kaggle.com/datasets/nyahmet/fixed-wing-uav-dataset) | Ahmet Nuri Yılmaz (nyahmet) | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |
| [VisioDECT](https://www.kaggle.com/datasets/simeonajakwe/visiodect-for-scenario-based-multi-drone-detection) | Simeon Okechukwu Ajakwe, Vivian Ukamaka Ihekoronye, Golam Mohtasin, Rubina Akter, Dong Seong Kim and Jae Min Lee (DOI-listed creators) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), as listed by the Kaggle distribution used here |
| [MBAOD v1](https://data.mendeley.com/datasets/zfc8hndczp/1) | Firas Mohammed Kadhum, Abbas Abdulazeez Abdulhameed and Jameelah Harbi Saud | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Changes: source labels were remapped to the five RGB5 classes; images and boxes were resized with padding to 640 × 640; candidates were filtered, grouped and balanced as described in README.md. Adapted VisioDECT material is distributed under CC BY-NC-SA 4.0. Preserve its attribution and noncommercial/share-alike terms. The other sources retain their stated licenses. `manifest.jsonl` identifies the source of each image and records final image/label checksums.

The public manifest removes machine-specific absolute paths and adds checksums for the copied files. `summary.json` preserves the original training-manifest fingerprint for the recorded experiments.

Creator/license verification: Kaggle's public metadata identifies nyahmet as Ahmet Nuri Yılmaz and lists CC0; the used VisioDECT distribution lists CC BY-NC-SA 4.0. The original VisioDECT DOI [10.21227/n27q-7e06](https://doi.org/10.21227/n27q-7e06) lists the six creators above and a CC BY 4.0 notice in its [DataCite metadata](https://api.datacite.org/dois/10.21227/n27q-7e06). This package retains the more restrictive terms recorded for its Kaggle source rather than silently substituting a different distribution's license. The original 2022 dataset citation also credits Ali Aouto; that additional credit is retained here, as recorded in the dataset creators' [2026 data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12877808/).

HusAero curation, remapping, integration, fine-tuning and evaluation: **Hussain Alrashidi (27kwt)**. This attribution applies to original project contributions; it does not transfer source-image ownership or replace any upstream license.
