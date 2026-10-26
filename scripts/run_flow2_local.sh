#!/bin/bash
set -e

# Đảm bảo đang ở thư mục gốc của dự án
cd "$(dirname "$0")/.."

mkdir -p results

echo "🚀 Bắt đầu khởi chạy huấn luyện Flow 2 (MultiFrameSRResTran) trên Local MPS..."
export PYTHONUNBUFFERED=1

# Khởi chạy trong nền với nohup, ghi log chi tiết
nohup .venv/bin/python3 train.py \
  -m sr_restran \
  -n sr_restran_flow2 \
  --data-root dataset/data/train \
  --warmstart-ocr results/restran_kaggle_gpu_best.pth \
  --warmstart-sr results/sr_net_best.pth \
  --epochs 30 \
  --batch-size 64 \
  --lr 3e-4 \
  --output-dir results \
  > results/train_flow2_local.log 2>&1 &

PID=$!
echo $PID > results/flow2_local_pid.txt

echo "✅ Tiến trình huấn luyện Flow 2 đã được kích hoạt thành công!"
echo "📌 PID: $PID (đã lưu tại results/flow2_local_pid.txt)"
echo "📝 Log file: results/train_flow2_local.log"
