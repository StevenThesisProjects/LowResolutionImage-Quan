#!/bin/bash
set -e

echo "🚀 Chuẩn bị Kaggle Kernel cho Flow 2 (ResTran) Two-Stage Joint Training..."

mkdir -p kaggle_push_kernel
cd kaggle_push_kernel

cat << 'JSON_EOF' > kernel-metadata.json
{
  "id": "dotrongminhquan/lrlpr-flow-2-joint-restran-v13",
  "title": "lrlpr-flow-2-joint-restran-v13",
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

print("="*60)
print("🚀 LRLPR FLOW 2: MULTIFRAME SR-RESTRAN TRAINING (ACTION 2)")
print("="*60)

print("📦 1. Khôi phục mã nguồn từ dataset...")
zip_paths = glob.glob('/kaggle/input/**/codebase.zip', recursive=True)
if zip_paths:
    print(f"   Unzipping {zip_paths[0]}...")
    os.system(f"unzip -o -q {zip_paths[0]} -d /kaggle/working/")
else:
    codebase_paths = glob.glob('/kaggle/input/**/train.py', recursive=True)
    if codebase_paths:
        codebase_dir = os.path.dirname(codebase_paths[0])
        print(f"   Tìm thấy mã nguồn tại: {codebase_dir}")
        os.system(f"cp -r {codebase_dir}/* /kaggle/working/")
    else:
        print("⚠️ Quét toàn bộ tìm codebase.zip...")
        os.system("find /kaggle/input -name 'codebase.zip' -exec unzip -o -q {} -d /kaggle/working/ \;")

os.system("mkdir -p /kaggle/working/data")

print("🔗 2. Liên kết dữ liệu huấn luyện và kiểm định...")
kaggle_train_paths = glob.glob('/kaggle/input/**/train', recursive=True)
if kaggle_train_paths:
    train_dir = kaggle_train_paths[0]
    os.system(f"ln -sfn {train_dir} /kaggle/working/data/train")
    print(f"   ✅ Linked train data: {train_dir} -> /kaggle/working/data/train")
else:
    print("❌ LỖI: Không tìm thấy thư mục train trong /kaggle/input!")
    sys.exit(1)

kaggle_test_paths = glob.glob('/kaggle/input/**/public_test', recursive=True) or glob.glob('/kaggle/input/**/*test*', recursive=True)
if kaggle_test_paths:
    test_dir = kaggle_test_paths[0]
    os.system(f"ln -sfn {test_dir} /kaggle/working/data/public_test")
    print(f"   ✅ Linked test data: {test_dir} -> /kaggle/working/data/public_test")

val_track_paths = glob.glob('/kaggle/input/**/val_tracks.json', recursive=True)
if val_track_paths:
    os.system(f"cp {val_track_paths[0]} /kaggle/working/data/val_tracks.json")
    print(f"   ✅ Copied val_tracks.json: {val_track_paths[0]}")

print("📦 3. Cài đặt thư viện phụ thuộc...")
os.system("pip install -q albumentations")

print("🔥 4. BẮT ĐẦU HUẤN LUYỆN FLOW 2 (V13 WARMSTART + FREEZE-SR): train.py -m sr_restran...")
os.chdir("/kaggle/working")
os.system("mkdir -p results")

train_cmd = [
    "python", "-u", "train.py",
    "-m", "sr_restran",
    "--freeze-sr",
    "--checkpoint", "results/flow2_warmstart_init.pth",
    "--pretrained-backbone",
    "--epochs", "25",
    "--lr", "3e-4",
    "--batch-size", "64",
    "--num-workers", "2",
    "--max-hours", "2.5",
    "--require-gpu"
]
print("Lệnh chạy:", " ".join(train_cmd))
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
ret = subprocess.run(train_cmd, env=env)

if ret.returncode != 0:
    print("❌ Training failed with exit code:", ret.returncode)
    sys.exit(ret.returncode)

print("\n" + "="*60)
print("📊 5. ĐÁNH GIÁ MÔ HÌNH VỚI evaluate.py...")
print("="*60)
eval_cmd = [
    "python", "evaluate.py",
    "-m", "sr_restran",
    "--weights", "results/sr_restran_best.pth",
    "--output-json", "results/evaluation_metrics_sr_restran.json"
]
print("Lệnh đánh giá:", " ".join(eval_cmd))
subprocess.run(eval_cmd)

print("\n🎉 HOÀN THÀNH HUẤN LUYỆN VÀ ĐÁNH GIÁ FLOW 2!")
os.system("ls -la results/")
PY_EOF

echo "☁️ Đang push Kaggle Kernel v13..."
yes | ../.venv/bin/kaggle kernels push -p .

echo "🎉 Xong! Kernel Flow 2 đã được đẩy lên Kaggle."
echo "🔗 Link: https://www.kaggle.com/dotrongminhquan/lrlpr-flow-2-joint-restran-v13"
