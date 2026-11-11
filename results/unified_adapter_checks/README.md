# Unified output adapter: functional verification

verification.json records eight actual CPU image-inference checks: one test image from each of the four source datasets, processed by each unchanged native RGB5 checkpoint. Native classes and PT bytes remained unchanged. Mapping, overlap handling, default non-drone abstention and low/conflicting-score abstention were checked.

This is a software smoke check, not an accuracy/mAP benchmark, a four-category training run, a particular aircraft recognition test or a new Raspberry Pi deployment. Existing image and video measurements remain under results/rgb5_v2 with their original native class maps and protocols. Unknown is a conservative display decision, not a learned class or a guarantee that a high-scoring model prediction is correct.
