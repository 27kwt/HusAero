# Earlier experiment versions

Current data is in [dataset](../dataset/). Current native four-class weight details (binary upload pending) are in [models/rgb4](../models/rgb4/). Earlier RGB5 checkpoints, HEFs and measured results remain in [models/rgb5_v2](../models/rgb5_v2/) and [results/rgb5_v2](../results/rgb5_v2/); their metric files and model bytes have not been changed.

For the exact original dataset layout, labels, scripts and generic five-class release, use [the frozen repository snapshot](https://github.com/27kwt/HusAero/tree/774b59ec94fbd4520f13b0bd019c44cb99d4c33d). Original RGB5 labels are [here](https://github.com/27kwt/HusAero/tree/774b59ec94fbd4520f13b0bd019c44cb99d4c33d/dataset/combined_rgb5_v2). The older starter dataset is preserved in that snapshot rather than mixed into the current dataset.

To reproduce historical RGB5 evaluation, check out that exact commit into a separate folder. Its `combined_rgb5_v2/images` and `combined_rgb5_v2/labels` must remain paired. Do not point an RGB5 evaluation at today's four-class labels.

Obsolete generic-release files are omitted from the current project. The extra class has been removed from the current dataset, demonstration samples and runtime defaults. Historical results keep their original meanings, and Git history is preserved.
