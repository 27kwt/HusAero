# RGB5 v2 data card

Maintainer and HusAero integration: Hussain Alrashidi (27kwt), 2026.
Purpose: experimental RGB detection of fixed-wing drones, multirotors, birds, airplanes and helicopters.
Size: 19,536 images with YOLO labels; 13,954 train / 3,423 validation / 2,159 test. Input: 640 × 640 RGB, resized with padding. No IR/thermal imagery.

The four sources, remapping, grouping, exclusions and limitations are documented in README.md. Per-image source, original filename, group, split and final file hashes are recorded in manifest.jsonl. Public manifest paths use forward slashes. summary.json retains the original training-manifest fingerprint; it is not the SHA-256 of the sanitized public manifest.

Source photos are public datasets, not footage recorded by the HusAero team. No exhaustive visual re-annotation of every object has been performed. Structural and checksum audit: FILE_AUDIT.json. Recorded groups do not cross partitions; complete independence of recording sessions cannot be established from all source metadata. Starting checkpoints have earlier, different training histories.

All 2,159 test images were evaluated on PC after validation-based checkpoint selection. A 500-image stratified sample is for a separate hardware evaluation, not additional test images and not a population-balanced estimate.

Known limitation: both RGB5 candidates misclassified a visible multirotor as fixed-wing in an independent video frame. Public-image metrics do not establish dependable subtype recognition in new scenes. These models are experimental.

Attribution and reuse terms: LICENSE_DATA.md. VisioDECT adaptations retain CC BY-NC-SA 4.0 terms; other sources retain their stated licenses. HusAero's software license does not replace image licenses or claim ownership of upstream content.
