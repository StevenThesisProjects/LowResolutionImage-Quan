#!/bin/bash
set -e

# Đảm bảo đang ở thư mục gốc của dự án
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p results

echo "=========================================================================="
echo "🚀 Khởi chạy Huấn luyện Flow 2 v2.2 (MultiFrameSRResTran) 12 Epochs trên Local MPS"
echo "=========================================================================="
export PYTHONUNBUFFERED=1

nohup "$PROJECT_ROOT/.venv/bin/python" train.py \
  -m sr_restran \
  --checkpoint results/sr_restran_flow2_plus10_best.pth \
  --experiment-name sr_restran_flow2_v22 \
  --epochs 12 \
  --batch-size 64 \
  --lr 1.0e-4 \
  --ocr-lr-scale 0.8 \
  --aug-level light \
  --use-hard-loss \
  --data-root dataset/data/train \
  --val-split-file dataset/data/val_tracks.json \
  --output-dir results \
  > results/train_flow2_v22.log 2>&1 &

PID=$!
echo $PID > results/flow2_v22_pid.txt

echo "✅ Tiến trình huấn luyện Flow 2 v2.2 đã được kích hoạt thành công!"
echo "📌 PID: $PID (đã lưu tại results/flow2_v22_pid.txt)"
echo "📝 Log file: results/train_flow2_v22.log"
