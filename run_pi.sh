#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ "$#" -lt 2 ]; then echo "Usage: bash run_pi.sh /path/to/rgb_sequences /path/to/new_output [HusAero-8n|HusAero-11n]"; exit 2; fi
model="${3:-HusAero-8n}"
case "$model" in HusAero-8n|HusAero-11n) ;; *) echo "Unknown model"; exit 2;; esac
hef="models/rgb4/${model}_RGB4.hef"
if [ ! -f "$hef" ]; then
  echo "RGB4 Hailo-8 HEF is not available yet. Compile and verify it before this run."
  echo "Earlier RGB5 hardware evidence and runtime: models/rgb5_v2/README.md"
  exit 2
fi
python_bin="${HusAero_PI_PYTHON:-python3}"
exec "$python_bin" deployment/hailo_follow_configured.py --hef "$hef" --tracking-config "configs/$model-follow.json" --video-root "$1" --out "$2" --save-videos
