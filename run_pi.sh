#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ "$#" -lt 2 ]; then echo "Usage: bash run_pi.sh /path/to/rgb_sequences /path/to/new_output [YOLOKW8n|YOLOKW11n]"; exit 2; fi
model="${3:-YOLOKW8n}"
case "$model" in YOLOKW8n|YOLOKW11n) ;; *) echo "Unknown model"; exit 2;; esac
python_bin="${YOLOKW_PI_PYTHON:-python3}"
exec "$python_bin" deployment/hailo_follow_configured.py --hef "models/$model.hef" --tracking-config "configs/$model-follow.json" --video-root "$1" --out "$2" --save-videos
