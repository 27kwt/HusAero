# HusAero

RGB aerial-object detection and drone tracking with **HusAero-8n** (based on
Ultralytics YOLOv8n) and **HusAero-11n** (based on Ultralytics YOLO11n).

The models detect `drone`, `bird`, `airplane`, `helicopter` and `kite`.
Multirotor and fixed-wing UAVs share the `drone` class. The software does not
classify civilian/military status or drone subtypes.

**Results:** [evaluation report](results/HusAero_Results.pdf),
[machine-readable metrics](results/metrics.json),
[test-image predictions](results/test_images), and
[per-epoch CSV records](results/training).

| Recorded Pi 5 + Hailo-8 measurement | HusAero-8n | HusAero-11n |
|---|---:|---:|
| Public 250-image mAP50-95 | 72.81% | 76.71% |
| Drone AP50-95 on the same image test | 40.40% | 46.28% |
| Follow localization on 16,286 target-present video frames | 78.79% | 61.30% |
| Follow output on 856 source-target-absent frames | 121 | 55 |
| Recorded-file decode + inference + follow + logs FPS | 56.56 | 44.07 |

HusAero-8n is the selected Pi/video default. HusAero-11n remains available for its
stronger image-box AP. Throughput excludes drawing and video encoding and is
not a live-camera latency measurement. The models have different prior training
histories; the final scores do not isolate architecture alone.

## 1. Download the runnable package

Open this repository's **Releases** and download
`HusAero-v0.3.0-rc5-Complete.zip`. Extract it and enter the `HusAero` folder.
It contains both `.pt` checkpoints, both tested Hailo-8 `.hef` files, all code,
20 labeled samples, and the 1,700-image starter dataset.

The source repository and GitHub's automatic source-code archives omit model
binaries and the full dataset. Use the Complete asset for immediate execution.
Model file hashes are recorded in `models/registry.json`.

## 2. Install and run detection on a PC

Use Python 3.11. Initial dependency installation needs internet access.
The following Windows commands are for **Command Prompt**:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python verify_release.py --full
python demo.py --model both
```

On Linux, create and activate the environment with `python3.11 -m venv .venv`
and `source .venv/bin/activate`, then run the same `python` commands.
Windows users can alternatively run `setup_cpu.cmd` and then `run_demo.cmd`.

```text
python demo.py --model HusAero-8n --source /path/to/images
python demo.py --model HusAero-11n --source /path/to/video.mp4
```

Outputs go into a new timestamped `runs/demo_*` directory. Images/videos contain
model predictions, without a ground-truth overlay. Box scores are prediction
confidence, not the accuracy of the whole system. The 20 samples contain four
images per class and belong to the public test split; they are a smoke test,
not an additional independent benchmark.

## 3. Evaluate the labeled test split

```text
python evaluate_samples.py --model HusAero-8n --data dataset/data.yaml --device cpu
python evaluate_samples.py --model HusAero-11n --data dataset/data.yaml --device cpu
```

This uses Ultralytics evaluation and saves metrics and diagnostic plots under
`runs/`. Its results can differ from the recorded Hailo results, which use a
shared NumPy 101-point AP scorer. To use that scorer for PC weights:

```text
python pc_detection_eval.py --weights models/HusAero-8n.pt --data dataset --device cpu --out runs/pc_shared_8n
python pc_detection_eval.py --weights models/HusAero-11n.pt --data dataset --device cpu --out runs/pc_shared_11n
```

Use a fresh output directory each time. Precision measures how many predicted
boxes match labels; recall measures how many labeled objects were found. mAP50
uses IoU 0.5; mAP50-95 averages AP over IoU thresholds 0.5 through 0.95.
These detection metrics are separate from single-drone video localization.

## 4. Dataset structure and adding images

The public starter set is derived from MBAOD v1:

| Split | Images | Images per class |
|---|---:|---:|
| Train | 1,200 | 240 |
| Validation | 250 | 50 |
| Test | 250 | 50 |

```text
dataset/
  data.yaml
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

Each image has a matching `.txt` label filename. Every label row contains
`class_id x_center y_center width height`, with coordinates normalized to
0-1. Class IDs are `0 drone`, `1 bird`, `2 airplane`, `3 helicopter`, `4 kite`.
Parachutes/paragliders are excluded. Do not train on the saved prediction
images in `results/test_images`; those already contain rendered boxes.

To add data, preserve source images, record source permission and provenance,
label each visible object, and create a new dataset directory. Split whole
recording sessions before selecting frames; keep similar frames from the same
recording in one split. Review labels and background-only images. Use validation
for checkpoint/threshold selection, and reserve the test split for final scoring.
`prepare_video_frames.py --help` describes the explicit train/val/reserved-group
RGB extraction interface. It uses source annotation files and does not draw
boxes into training frames.

The starter selection used image-quality and perceptual duplicate screening;
`dataset/curation_report.json` and the Complete inventory record the process.
Source-session independence and exhaustive semantic annotation accuracy are
not established. The starter set is not the complete training data of the
released checkpoints. Their adaptation also used private DUT images and
source-video frames, which are not redistributed.

## 5. Train or continue fine-tuning

```text
python finetune.py --model HusAero-8n --data dataset/data.yaml --epochs 10 --batch 4 --device cpu
python finetune.py --model HusAero-11n --data dataset/data.yaml --epochs 10 --batch 4 --device cpu
```

For a configured CUDA environment, use `--device 0` and an appropriate batch
size. CPU training is supported but slower. The script resolves `data.yaml`,
loads release weights, trains on the training split, validates each epoch,
and saves `best.pt`, `last.pt`, `results.csv`, configuration and plots in a new
run. It starts a new optimizer and does not overwrite release weights or resume
every historical optimizer state. These example runs are additional training,
not an exact replay of the private adaptation experiment.

The completed adaptation used 2,219 training and 721 validation images, 10
epochs per model, 640px, batch 16, SGD, initial LR 0.0003, seed 42 and identical
new augmentation settings. All settings are in `configs/adaptation_protocol.json`.
The selected checkpoints are epoch 8 for HusAero-8n and epoch 6 for HusAero-11n.
The earlier matched six-class study completed 50 epochs per architecture;
later five-class/rehearsal stages and the separate one-epoch Pi CPU experiment
have their own CSV records. These are separate stages, not one uninterrupted run.

For each epoch, `results.csv` stores train/validation box, classification and
DFL losses, plus validation precision, recall, mAP50 and mAP50-95.
The PDF contains the current curves. To regenerate plots from a saved run:

```text
python -m pip install -r requirements-analysis.txt
python plot_training.py --run results/training/adaptation10 --out runs/new_curves --models HusAero-8n HusAero-11n
```

Training/validation curves do not replace a held-out test. Additional epochs
are useful only when validation supports them.

## 6. Detection and tracking in video

Detection produces five-class boxes for each frame. The current Pi pipeline
then selects one drone with `FlowAssistedTracker`: gated detection association,
confirmation, and bounded Lucas-Kanade optical flow during brief detector gaps.
NMS and nested-drone-box suppression reduce duplicate outputs; they do not
guarantee no false boxes, missed targets or drift.

The optional **PC** diagnostic uses Ultralytics **ByteTrack**:

```text
python demo.py --model HusAero-8n --source /path/to/video.mp4 --track
```

With `--track`, `--conf` sets the activation confidence for a new track.
Lower-confidence detections are retained down to the saved ByteTrack low
threshold for association with existing tracks. Each run saves its effective
tracker settings and detector threshold.

### Optional RGB video example

The [official Anti-UAV repository](https://github.com/ZhaoJ9014/Anti-UAV#data-preparation)
links to the **Anti-UAV300** download. Choose the 300 version for RGB video;
the 410 and 600 versions contain infrared video only. Review the source terms
before using its data. The external download is a dataset archive, not a
small video bundled with this release.

After downloading and extracting it, locate
`test/20190925_111757_1_10/visible.mp4`. From the extracted HusAero Complete
folder, run the following, replacing the example path with your local path:

```text
python demo.py --model HusAero-8n --source "C:/Anti-UAV/test/20190925_111757_1_10/visible.mp4" --track
```

This RGB clip is approximately 17.4 seconds at 20 FPS. Use `visible.mp4`,
not `infrared.mp4`. The demo saves the prediction video, per-frame boxes,
tracker settings and summary under a new `runs/demo_*` directory. No ground-truth
boxes are drawn. This example demonstrates the PC ByteTrack path, not the Pi
FlowAssistedTracker benchmark. It is not a new independent accuracy test.

Anti-UAV source footage and annotations remain credited to their original
authors. The video is not redistributed here; obtain it from the source.
You can use your own authorized RGB `.mp4` instead.

This is a different tracker from the current Pi follow pipeline. For a shared-
detection SORT/ByteTrack comparison, install `requirements-analysis.txt` and use
`compare_trackers.py --help`. Its source format is sequence/visible.mp4 and
visible.json; annotations are used only by the scorer. No formal multi-object
IDF1/HOTA claim is made from these single-target diagnostics.

The supplied source videos and extracted frames are not included in the public
assets because redistribution permission is unverified. Use your own authorized
RGB video. Runtime creates prediction videos; no GT enters the tracker. Detection
and tracking outputs are separate so a reference box is not mistaken for a
duplicate prediction.

## 7. Raspberry Pi 5 + Hailo-8

The supplied `.hef` files were tested on Pi 5 8GB with Hailo-8 and HailoRT/
firmware 4.23.0. They target **Hailo-8, not Hailo-8L**. Reuse a compatible
working Hailo Python environment; the PC pip requirements do not install the
vendor driver or hardware bindings. Training again on the Pi is not required.

Copy the Complete package into a new Pi directory. From the PC, replace
`user@raspberrypi.local` with your Pi login and hostname:

```text
scp HusAero-v0.3.0-rc5-Complete.zip user@raspberrypi.local:~/
ssh user@raspberrypi.local
```

On the Pi, extract into a new directory and enter the package:

```bash
mkdir HusAero-rc5
unzip HusAero-v0.3.0-rc5-Complete.zip -d HusAero-rc5
cd HusAero-rc5/HusAero
```

With the working Hailo interpreter activated:

```bash
python3 verify_release.py --full
python3 -m unittest deployment.test_husaero_decoder test_flow_assisted_tracker test_detection_scores -v
python3 deployment/husaero_hailo.py --hef models/HusAero-8n.hef --data dataset --split test --out runs/pi_images_8n
python3 deployment/husaero_hailo.py --hef models/HusAero-11n.hef --data dataset --split test --out runs/pi_images_11n
bash run_pi.sh /path/to/rgb_sequences /path/to/new_results HusAero-8n
```

The video root must contain `sequence-name/visible.mp4`. If `python3` is not the
working Hailo interpreter, set `HusAero_PI_PYTHON` to its executable path.
`detections.mp4` shows raw detections; `follow.mp4` shows one selected drone.
Green means a fresh detection association; yellow means bounded flow/motion
propagation; SEARCHING/LOST means no confirmed track. The source camera reticle
is part of input footage. Camera capture and pan-tilt motor control have not
been validated by the recorded-file tests.

For export of newly trained weights:

```text
python -m pip install -r requirements-analysis.txt
python export_onnx.py --weights /path/to/best.pt --out runs/new_export
python create_calibration.py --data dataset/data.yaml --out runs/new_calibration --count 1024
```

On a separate Linux x86_64 Hailo DFC 3.34 environment:

```bash
python compile_hailo.py --onnx /path/to/model.onnx --out /path/to/new_compile --name HusAero-8n --calibration /path/to/calibration.npy --class-names drone bird airplane helicopter kite
```

Calibration must use training images only. Compilation is separate from Pi
inference; use the supplied tested HEFs to skip it. Newly compiled models need
new verification and hardware evaluation.

Calibration manifests store paths relative to the dataset root. Copy the whole
dataset when replaying a selection on another computer. External image paths
must be materialized inside that dataset root first.

## Release provenance

HusAero is the new project name for the previously published YOLOKW package.
Version v0.3.0-rc5 renames files and commands; the pretrained PT/HEF bytes,
training histories and measured scores are unchanged. Embedded checkpoint
metadata may retain historical names. This is not a new training run or
new hardware benchmark. Earlier validation records refer to earlier releases.

## Reuse and limits

Read [NOTICE.md](NOTICE.md) and the included [LICENSE](LICENSE). Keep upstream
credits, dataset provenance and visible source watermarks. These are fine-tuned
YOLO models, not new YOLO architectures or original project photography.
The report includes the observed recall/false-output tradeoff, per-class image
scores, previous training histories and unverified source-session independence.
The package is an experimental research release, not a zero-error field system.

The baseline software was checked in a fresh Python 3.11 CPU environment.
The rc4 review adds targeted regression checks for threshold-first precision/
recall, ByteTrack low-score association, portable calibration manifests,
evaluation image counts, and Git/download integrity. Saved 250-image CPU
predictions per model were rescored without changing model weights or the
historical AP calculation. See `results/software_validation.json` for the
tested versions and scope; plumbing checks are not accuracy benchmarks or
new hardware tests.
