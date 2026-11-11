# Current dataset: unified drone / RGB4

Fixed-wing and multirotor drones now share **drone**. There is no kite class and no claimed aircraft mission or subtype recognition.

| ID | Class |
| --- | --- |
| 0 | drone |
| 1 | bird |
| 2 | airplane |
| 3 | helicopter |

| Split | Images | YOLO labels |
| --- | ---: | ---: |
| Train | 13,954 | 13,954 |
| Validation | 3,423 | 3,423 |
| Test | 2,159 | 2,159 |
| Total | 19,536 | 19,536 |

All RGB images are already in GitHub under [the shared image folder](../combined_rgb5_v2/images/). The remapped labels are included here in labels/. The same image files, 640×640 dimensions, source groups and partitions are retained. Bounding-box coordinates are unchanged. Image counts by class and object counts are in summary.json; multi-object images can contribute to more than one class.

From the repository root, prepare YOLO's matching image/label layout once:

```sh
python tools/prepare_current_dataset.py
```

This checks every image and label checksum and creates local image links (or copies when links are unavailable). It does not download data, train a model, alter source images or change the split. Local images/ entries are ignored by Git because their bytes are already stored in the shared image folder. Use --copy-images if you need independent image copies. Treat linked source images as read-only.

Then use dataset/combined_rgb4/data.yaml for a future four-category training run. Do not evaluate native five-output checkpoints against these four-category IDs without the output adapter. No four-category training or merged mAP measurement has been performed by this relabeling step.

Sources are Yaren Koç and Ahmet Nuri Yılmaz (fixed-wing inputs), VisioDECT (multirotor inputs), and MBAOD v1 (birds, airplanes, helicopters). These are four source datasets, grouped into three uses before unification. DUT Anti-UAV belongs to earlier checkpoint training/video provenance, not these 19,536 source images. Preparation and grouping details remain in [the original data card](../combined_rgb5_v2/DATA_CARD.md); [licenses and attribution](LICENSE_DATA.md) are preserved. manifest.jsonl records each image source and original/new label fingerprints.

Existing best HusAero checkpoints internally retain the old five outputs. tools/predict_unified.py remaps their predictions to these four categories without pretending to retrain the network. Its conservative default display uses drone or unknown_airborne_object; raw model hypotheses remain in the output log. High scores do not prove correct identity, and the adapter cannot guarantee detection of every unfamiliar drone shape.
