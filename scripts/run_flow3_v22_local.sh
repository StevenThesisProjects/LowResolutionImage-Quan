#!/bin/bash
set -e

# Đảm bảo đang ở thư mục gốc của dự án
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p results

echo "=========================================================================="
echo "🚀 Khởi chạy Huấn luyện Flow 3 v2.2 (MultiFrameSRViT) 12 Epochs trên Local MPS"
echo "=========================================================================="
export PYTHONUNBUFFERED=1

nohup caffeinate -dimsu "$PROJECT_ROOT/.venv/bin/python" train.py \
  -m sr_vit \
  --checkpoint results/sr_vit_flow3_best.pth \
  --experiment-name sr_vit_flow3_v22 \
  --epochs 12 \
  --batch-size 64 \
  --lr 1.2e-4 \
  --ocr-lr-scale 1.0 \
  --aug-level light \
  --use-hard-loss \
  --grad-clip 1.0 \
  --data-root dataset/data/train \
  --val-split-file dataset/data/val_tracks.json \
  --output-dir results \
  > results/train_flow3_v22.log 2>&1 &

PID=$!
echo $PID > results/flow3_v22_pid.txt

echo "✅ Tiến trình huấn luyện Flow 3 v2.2 đã được kích hoạt thành công!"
echo "📌 PID: $PID (đã lưu tại results/flow3_v22_pid.txt)"
echo "📝 Log file: results/train_flow3_v22.log"
