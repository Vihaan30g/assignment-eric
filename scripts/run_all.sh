#!/bin/bash
cd "$(dirname "$0")/.."

run() {   # usage: run <pretrained .pt> <name>
  if [ -f "results/training/$2/test_metrics.json" ]; then
    echo "== $2 already finished, skipping"
  elif [ -f "runs/$2/weights/last.pt" ]; then
    echo "== $2 interrupted, resuming"
    python src/train.py --model "$1" --name "$2" --resume
  else
    python src/train.py --model "$1" --name "$2"
  fi
}

run yolo11n.pt y11n
run yolo11s.pt y11s