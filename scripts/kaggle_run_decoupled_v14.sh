#!/bin/bash
set -e

echo "=========================================================================="
echo "🚀 Chuẩn bị và Đẩy Kaggle Kernel: Flow 2 Decoupled Two-Stage Training (v14)"
echo "=========================================================================="

# 1. Sync updated codebase to Kaggle Dataset
echo "📦 Bước 1: Đồng bộ Codebase lên Kaggle Dataset (dotrongminhquan/lrlpr-codebase)..."
bash scripts/kaggle_sync.sh

# 2. Chuẩn bị Kernel Metadata & Template
echo "📦 Bước 2: Tạo thư mục cấu hình kernel kaggle_push_kernel..."
mkdir -p kaggle_push_kernel
cd kaggle_push_kernel

cat << 'JSON_EOF' > kernel-metadata.json
{
  "id": "dotrongminhquan/lrlpr-flow-2-decoupled-v14",
  "title": "lrlpr-flow-2-decoupled-v14",
  "code_file": "kaggle_kernel_template.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": [
    "qunnguynm/lrlpr-2026",
    "dotrongminhquan/lrlpr-codebase"
  ],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
JSON_EOF

cat << 'PY_EOF' > kaggle_kernel_template.py
import os
import sys
import subprocess
import glob
import time

print("=" * 70)
print("🚀 KAGGLE GPU: BƯỚC 3 (TWO-STAGE DECOUPLED TRAINING & JOINT FINE-TUNING)")
print("=" * 70)

print("\n📦 1. Khôi phục mã nguồn từ dataset...")
zip_paths = glob.glob('/kaggle/input/**/codebase.zip', recursive=True)
if zip_paths:
    print(f"   Unzipping {zip_paths[0]} to /kaggle/working/...")
    os.system(f"unzip -o -q {zip_paths[0]} -d /kaggle/working/")
else:
    print("❌ Không tìm thấy codebase.zip! Thử tìm file train_decoupled.py...")
    codebase_paths = glob.glob('/kaggle/input/**/train_decoupled.py', recursive=True)
    if codebase_paths:
        codebase_dir = os.path.dirname(codebase_paths[0])
        print(f"   Copy từ: {codebase_dir}")
        os.system(f"cp -r {codebase_dir}/* /kaggle/working/")
    else:
        print("❌ LỖI NGHIÊM TRỌNG: Không tìm thấy mã nguồn!")
        sys.exit(1)

os.system("mkdir -p /kaggle/working/data /kaggle/working/results")

print("\n🔗 2. Liên kết dữ liệu huấn luyện và kiểm định...")
kaggle_train_paths = glob.glob('/kaggle/input/**/train', recursive=True)
if kaggle_train_paths:
    train_dir = kaggle_train_paths[0]
    os.system(f"ln -sfn {train_dir} /kaggle/working/data/train")
    print(f"   ✅ Linked train data: {train_dir} -> /kaggle/working/data/train")
else:
    print("❌ LỖI: Không tìm thấy thư mục train trong /kaggle/input!")
    sys.exit(1)

val_track_paths = glob.glob('/kaggle/input/**/val_tracks.json', recursive=True)
if val_track_paths:
    os.system(f"cp {val_track_paths[0]} /kaggle/working/data/val_tracks.json")
    print(f"   ✅ Copied val_tracks.json: {val_track_paths[0]}")

warm_paths = glob.glob('/kaggle/input/**/flow2_warmstart_init.pth', recursive=True)
if warm_paths:
    os.system(f"cp {warm_paths[0]} /kaggle/working/results/flow2_warmstart_init.pth")
    print(f"   ✅ Copied flow2_warmstart_init.pth: {warm_paths[0]}")
else:
    sr_paths = glob.glob('/kaggle/input/**/sr_net_best.pth', recursive=True)
    if sr_paths:
        os.system(f"cp {sr_paths[0]} /kaggle/working/results/sr_net_best.pth")
        print(f"   ✅ Copied sr_net_best.pth: {sr_paths[0]}")

print("\n📦 3. Cài đặt thư viện phụ thuộc...")
os.system("pip install -q albumentations")

print("\n🔥 4. BẮT ĐẦU HUẤN LUYỆN BƯỚC 3: train_decoupled.py...")
os.chdir("/kaggle/working")

train_cmd = [
    "python", "-u", "train_decoupled.py",
    "--model", "restran",
    "--frozen-epochs", "5",
    "--joint-epochs", "5",
    "--samples-per-epoch", "8192",
    "--batch-size", "32",
    "--lr-ocr-stage1", "1.5e-4",
    "--lr-ocr-stage2", "5e-5",
    "--lr-sr-stage2", "1e-5",
    "--lambda-sr", "0.1",
    "--num-workers", "2",
    "--max-hours", "1.0",
    "--init-weights", "results/flow2_warmstart_init.pth",
    "--output-dir", "results"
]

print("Lệnh chạy:", " ".join(train_cmd))
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
ret = subprocess.run(train_cmd, env=env)

if ret.returncode != 0:
    print(f"❌ Training kết thúc với mã lỗi: {ret.returncode}")
    sys.exit(ret.returncode)

print("\n" + "=" * 70)
print("📊 5. ĐÁNH GIÁ MÔ HÌNH VỚI evaluate.py...")
print("=" * 70)
best_ckpt = "results/decoupled_restran_best.pth"
if not os.path.exists(best_ckpt):
    best_ckpt = "results/decoupled_restran_last.pth"

eval_cmd = [
    "python", "evaluate.py",
    "-m", "sr_restran",
    "--weights", best_ckpt,
    "--output-json", "results/evaluation_metrics_decoupled.json"
]
print("Lệnh đánh giá:", " ".join(eval_cmd))
subprocess.run(eval_cmd)

print("\n🎉 HOÀN THÀNH TOÀN DIỆN BƯỚC 3 TRÊN KAGGLE GPU!")
os.system("ls -la results/")
PY_EOF

echo "☁️ Đang push Kaggle Kernel v14..."
yes | ../.venv/bin/kaggle kernels push -p .

cd ..
echo "🎉 Xong! Kernel Flow 2 Decoupled v14 đã được đẩy lên Kaggle."
echo "🔗 Link theo dõi: https://www.kaggle.com/dotrongminhquan/lrlpr-flow-2-decoupled-v14"
