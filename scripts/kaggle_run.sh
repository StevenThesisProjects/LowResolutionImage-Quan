#!/bin/bash

# Exit on error
set -e

echo "🚀 Chuẩn bị Kaggle Kernel để tự động hóa huấn luyện..."

KERNEL_DIR="kaggle_push/notebook"
rm -rf "$KERNEL_DIR"
mkdir -p "$KERNEL_DIR"

# Tạo kernel-metadata.json
cat <<EOF > "$KERNEL_DIR/kernel-metadata.json"
{
  "id": "dotrongminhquan/lrlpr-flow-2-decoupled-v10",
  "title": "lrlpr-flow-2-decoupled-v10",
  "code_file": "train_runner.py",
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
  "kernel_sources": []
}
EOF

# Tạo train_runner.py
cat <<'EOF' > "$KERNEL_DIR/train_runner.py"
import os
import sys

print("=== LRLPR KAGGLE RUNNER ===")

# Copy codebase to working directory (Kaggle runs scripts in /kaggle/working)
print("📦 Copying codebase...")
print("📦 Locating codebase...")
os.system("find /kaggle/input -name 'train_decoupled.py' -exec dirname {} \; > /tmp/codebase_dir.txt")
with open("/tmp/codebase_dir.txt", "r") as f:
    codebase_dir = f.read().strip()

if codebase_dir:
    print(f"Codebase found at {codebase_dir}")
    os.system(f"cp -r {codebase_dir}/* /kaggle/working/")
else:
    print("Codebase dir not found, attempting unzip...")
    os.system("unzip -o -q /kaggle/input/*/codebase.zip -d /kaggle/working/")

# Setup data paths
import glob
print("🔗 Linking dataset using glob search...")
os.system("mkdir -p /kaggle/working/data")

kaggle_train_paths = glob.glob('/kaggle/input/**/train', recursive=True)
if kaggle_train_paths:
    train_dir = kaggle_train_paths[0]
    os.system(f"ln -sfn {train_dir} /kaggle/working/data/train")
    print(f"✅ Linked {train_dir} -> data/train")
else:
    print("⚠️ Could not find train directory in /kaggle/input/")

kaggle_test_paths = glob.glob('/kaggle/input/**/public_test', recursive=True) or glob.glob('/kaggle/input/**/*test*', recursive=True)
if kaggle_test_paths:
    test_dir = kaggle_test_paths[0]
    os.system(f"ln -sfn {test_dir} /kaggle/working/data/public_test")
    print(f"✅ Linked {test_dir} -> data/public_test")
else:
    print("⚠️ Could not find public_test directory in /kaggle/input/")

# Verify data link
print("Data directory contents:")
os.system("ls -la /kaggle/working/data")

# Run the training
print("🚀 Starting Two-Stage Decoupled Training on GPU...")
os.chdir("/kaggle/working")
# The script will output to /kaggle/working/results, which Kaggle automatically saves
ret = os.system("python train_decoupled.py")
if ret != 0:
    print("❌ Training failed!")
    sys.exit(1)

print("✅ Training completed successfully!")
EOF

echo "☁️ Đang push Kaggle Kernel..."
cd "$KERNEL_DIR"
../../.venv/bin/kaggle kernels push

echo "
🎉 Xong! Kernel huấn luyện đã được đẩy lên Kaggle và đang chạy trên GPU.
🔗 Link theo dõi quá trình chạy:
https://www.kaggle.com/dotrongminhquan/lrlpr-training-decoupled
"
