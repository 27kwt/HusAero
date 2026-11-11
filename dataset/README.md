# Dataset files

The images and matching YOLO labels are included directly in this repository.

| Split | Images | Labels | Images per class |
| --- | ---: | ---: | ---: |
| Training | 1,200 | 1,200 | 240 |
| Validation | 250 | 250 | 50 |
| Test | 250 | 250 | 50 |

Class IDs in `data.yaml` are 0 drone, 1 bird, 2 airplane, 3 helicopter and 4 kite.
Each image in `images/<split>/` has a matching text file in `labels/<split>/`.

This curated starter subset comes from MBAOD v1. Source authors are Firas Mohammed Kadhum,
Abbas Abdulazeez Abdulhameed and Jameelah Harbi Saud. The publisher lists CC BY 4.0.
Source and changes are described in [NOTICE.md](../NOTICE.md).
Dataset DOI: https://doi.org/10.17632/zfc8hndczp.1

The training histories of the released checkpoints also used DUT and video frames;
those additional data are described in the project report and are not included here.
