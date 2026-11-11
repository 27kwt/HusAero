HusAero RGB5 held-out test comparison
Fresh evaluation on the same 2,159 RGB test images, at image size 640, GPU RTX 4090.
Five classes: fixed_wing_drone, multirotor_drone, bird, airplane, helicopter.
Both best checkpoints come from completed 50-epoch runs. These are your fine-tuned HusAero models.

Model          Precision   Recall     mAP50      mAP50-95   Drone mean AP50
HusAero-8n        96.25%     94.72%     97.89%     71.57%     98.21%
HusAero-11n       95.31%     94.41%     97.58%     71.67%     97.32%

Verdict: HusAero-8n leads precision, recall, mAP50, and mean AP50 across the two drone classes in this test. HusAero-11n has a very small lead in mAP50-95. There is no clear overall winner across all metrics; 8n is the stronger candidate on the drone AP50 criterion here.

Test plots are confidence-threshold curves, not per-epoch test curves. Test data is evaluated after checkpoint selection. Training_reference contains the original actual per-epoch training/validation graph and CSV, separately labelled.
Precision/recall use Ultralytics' best mean-F1 operating point, not necessarily confidence 0.25.
Drone mean AP50 is the arithmetic mean of fixed-wing and multirotor AP50, not merged-class AP.

Limitations: these public-dataset scores do not establish reliable drone subtype recognition in new video. Both candidates called a visible multirotor fixed-wing in a separate unannotated video. Complete recording-session independence is not established; starting checkpoints have different prior histories. No new Raspberry Pi runtime test is claimed here.
