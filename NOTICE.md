# Sources and attribution

HusAero-8n and HusAero-11n are fine-tuned Ultralytics YOLOv8n and YOLO11n models.
Upstream architecture/software: https://github.com/ultralytics/ultralytics .
This release follows the included AGPL-3.0 license; retained component notices
remain applicable. The project contributes dataset curation, fine-tuning,
evaluation, tracking integration and Raspberry Pi/Hailo deployment.

Dataset: MBAOD v1, Firas Mohammed Kadhum, Abbas Abdulazeez Abdulhameed and
Jameelah Harbi Saud. DOI: 10.17632/zfc8hndczp.1.
https://data.mendeley.com/datasets/zfc8hndczp/1
Publisher-listed license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ .
Changes: subset selection, file renaming, class/YOLO label conversion, duplicate
and quality screening, and split packaging. Photos are credited to their source
creators. Visible watermarks remain. Publisher licensing does not independently
establish clearance of every underlying watermarked image. See the provenance
records in samples and the Complete dataset inventory.

Parent and adaptation training used DUT Anti-UAV and supplied source recordings:
https://github.com/wangdongdut/DUT-Anti-UAV . Those images, recordings and
extracted frames are excluded from the public package. They are described in
the report so that the included 1,700-image starter set is not mistaken for the
complete training provenance of the shipped weights.

SORT: https://github.com/abewley/sort . Original license, source provenance and
headless-library modifications are retained in third_party/sort.
ByteTrack: https://github.com/ifzhang/ByteTrack ; the optional PC demo uses the
Ultralytics implementation. The current Pi follow results instead use
FlowAssistedTracker, with OpenCV Lucas-Kanade optical flow. Hailo runtime and
compiler software must be obtained separately from the vendor.

## HusAero project contribution

Copyright (c) 2026 Hussain Alrashidi (27kwt), for original HusAero contributions.
HusAero project integration, dataset curation and remapping, model fine-tuning,
evaluation, tracking integration and deployment tooling are credited to Hussain
Alrashidi. This credit does not claim authorship of Ultralytics architectures,
upstream code, public-source photographs or source annotations. Original project
code contributions are distributed under the included AGPL-3.0 license alongside
the retained upstream notices.

RGB5 v2 adds four image sources: Yaren Koç fixed-wing UAV, Nyahmet fixed-wing UAV,
Simeon Okechukwu Ajakwe and collaborators' VisioDECT, and MBAOD v1 by Firas Mohammed
Kadhum, Abbas Abdulazeez Abdulhameed and Jameelah Harbi Saud. Source links, licenses
and modifications are recorded in dataset/combined_rgb5_v2/LICENSE_DATA.md.
Do not treat HusAero credit or the code license as replacing their image rights.

Additional RGB5 attribution detail: nyahmet is Ahmet Nuri Yılmaz. VisioDECT DOI-listed creators are Simeon Okechukwu Ajakwe, Vivian Ukamaka Ihekoronye, Golam Mohtasin, Rubina Akter, Dong Seong Kim and Jae Min Lee; Ali Aouto is additionally credited in the original dataset citation. Distribution-license details and source links are in dataset/combined_rgb5_v2/LICENSE_DATA.md.
