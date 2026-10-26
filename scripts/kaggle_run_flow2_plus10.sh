#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================================="
echo "🚀 Chuẩn bị và Đẩy Kaggle Kernel: Flow 2 Joint Training +10 Epochs (v15 Fast-Finder)"
echo "=========================================================================="

# Chuẩn bị Kernel Metadata & Template
echo "📦 Bước 1: Tạo thư mục cấu hình kernel kaggle_push_kernel..."
mkdir -p kaggle_push_kernel
cd kaggle_push_kernel

cat << 'JSON_EOF' > kernel-metadata.json
{
  "id": "dotrongminhquan/lrlpr-flow-2-kaggle-plus10",
  "title": "lrlpr-flow-2-kaggle-plus10",
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
import time

# Chống nuốt log trên Kaggle
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

print("=" * 70)
print("🚀 KAGGLE GPU: FLOW 2 (SR-RESTRAN) FINE-TUNING +10 EPOCHS (FAST-FINDER)")
print("=" * 70)

def find_in_kaggle(target_name, is_dir=False, max_depth=3, base_dir="/kaggle/input"):
    """Fast non-recursive pruned finder that never walks into deep 38,000 track folders."""
    for root, dirs, files in os.walk(base_dir):
        rel = os.path.relpath(root, base_dir)
        depth = 0 if rel == "." else len(rel.split(os.sep))
        if is_dir and target_name in dirs:
            return os.path.join(root, target_name)
        if not is_dir and target_name in files:
            return os.path.join(root, target_name)
        if depth >= max_depth:
            dirs.clear()  # Prune deep directory recursion (avoids walking 190,000 files!)
    return None

t0 = time.time()
print("\n📦 1. Khôi phục mã nguồn từ dataset...")
codebase_train_py = find_in_kaggle('train.py', is_dir=False, max_depth=3)
if codebase_train_py:
    codebase_root = os.path.dirname(codebase_train_py)
    print(f"   ✅ Found codebase at: {codebase_root} in {time.time() - t0:.2f}s")
    os.system(f"cp -rf {codebase_root}/* /kaggle/working/")
else:
    print("❌ LỖI NGHIÊM TRỌNG: Không tìm thấy train.py trong /kaggle/input!")
    sys.exit(1)

os.system("mkdir -p /kaggle/working/data /kaggle/working/results")

print("\n🔗 2. Liên kết dữ liệu huấn luyện và kiểm định...")
train_dir = find_in_kaggle('train', is_dir=True, max_depth=3)
if train_dir:
    os.system(f"ln -sfn {train_dir} /kaggle/working/data/train")
    print(f"   ✅ Linked train data: {train_dir} -> /kaggle/working/data/train")
else:
    print("❌ LỖI: Không tìm thấy thư mục train trong /kaggle/input!")
    sys.exit(1)

test_dir = find_in_kaggle('public_test', is_dir=True, max_depth=3)
if test_dir:
    os.system(f"ln -sfn {test_dir} /kaggle/working/data/public_test")
    print(f"   ✅ Linked test data: {test_dir} -> /kaggle/working/data/public_test")

val_tracks_path = find_in_kaggle('val_tracks.json', is_dir=False, max_depth=3)
if val_tracks_path:
    os.system(f"cp {val_tracks_path} /kaggle/working/data/val_tracks.json")
    print(f"   ✅ Copied val_tracks.json: {val_tracks_path}")

target_ckpt = "/kaggle/working/results/sr_restran_flow2_plus10_best.pth"
if not os.path.exists(target_ckpt):
    found_ckpt = find_in_kaggle('sr_restran_flow2_plus10_best.pth', is_dir=False, max_depth=4)
    if found_ckpt:
        os.system(f"cp {found_ckpt} {target_ckpt}")
        print(f"   ✅ Checkpoint copied from: {found_ckpt}")

if os.path.exists(target_ckpt):
    print(f"   ✅ Checkpoint verified ({os.path.getsize(target_ckpt):,} bytes) at {target_ckpt}")
else:
    print(f"❌ LỖI: Không tìm thấy checkpoint {target_ckpt}!")
    sys.exit(1)

print(f"⚡ Toàn bộ bước chuẩn bị hoàn tất trong {time.time() - t0:.2f} giây!")

print("\n📦 3. Cài đặt thư viện phụ thuộc...")
os.system("pip install -q albumentations")

print("\n🔥 4. BẮT ĐẦU HUẤN LUYỆN FLOW 2 (10 EPOCHS GPU): train.py -m sr_restran...")
os.chdir("/kaggle/working")

train_cmd = [
    "python", "-u", "train.py",
    "-m", "sr_restran",
    "--checkpoint", target_ckpt,
    "--experiment-name", "sr_restran_flow2_kaggle_plus10",
    "--epochs", "10",
    "--lr", "1.0e-4",
    "--ocr-lr-scale", "0.8",
    "--aug-level", "light",
    "--use-hard-loss",
    "--batch-size", "64",
    "--num-workers", "2",
    "--max-hours", "4.0",
    "--require-gpu"
]

print("Lệnh chạy:", " ".join(train_cmd))
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
ret = subprocess.run(train_cmd, env=env)

if ret.returncode != 0:
    print("❌ Training failed with exit code:", ret.returncode)
    sys.exit(ret.returncode)

print("\n" + "=" * 70)
print("📊 5. ĐÁNH GIÁ MÔ HÌNH VỚI evaluate.py...")
print("=" * 70)
eval_cmd = [
    "python", "-u", "evaluate.py",
    "-m", "sr_restran",
    "--weights", "results/sr_restran_flow2_kaggle_plus10_best.pth",
    "--data-dir", "data/train",
    "--batch-size", "32",
    "--output-json", "results/eval_flow2_kaggle_plus10_best.json"
]
print("Lệnh đánh giá:", " ".join(eval_cmd))
subprocess.run(eval_cmd, env=env)

print("\n🎉 HOÀN THÀNH HUẤN LUYỆN VÀ ĐÁNH GIÁ FLOW 2 TRÊN KAGGLE!")
os.system("ls -lh results/")
PY_EOF

echo "☁️ Đang push Kaggle Kernel..."
yes | "$PROJECT_ROOT/.venv/bin/kaggle" kernels push -p .

echo "🎉 Xong! Kernel Flow 2 đã được đẩy lên Kaggle."
echo "🔗 Link: https://www.kaggle.com/code/dotrongminhquan/lrlpr-flow-2-kaggle-plus10"
