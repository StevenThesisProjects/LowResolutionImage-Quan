#!/bin/bash
set -e

echo "🚀 Chuẩn bị Kaggle Kernel cho luồng Joint Training (Flow 3 - SR-ViT)..."

# 1. Đóng gói mã nguồn mới nhất (Bao gồm cả val_tracks.json)
zip -r codebase.zip src configs scripts data/val_tracks.json results/sr_net_best.pth results/decoupled_sr_vit_best.pth train_joint.py train_decoupled.py evaluate.py > /dev/null

# 2. Tạo template kernel metadata
cat << 'JSON_EOF' > kernel-metadata.json
{
  "id": "dotrongminhquan/lrlpr-flow-3-joint-vit-v11",
  "title": "lrlpr-flow-3-joint-vit-v11",
  "code_file": "scripts/kaggle_kernel_template.py",
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

# 3. Tạo file script chạy Kaggle (kế thừa môi trường dataset)
cat << 'PY_EOF' > scripts/kaggle_kernel_template.py
import os
import subprocess
import glob

print("📦 Tự động giải nén mã nguồn mới nhất...")
import glob
import os
import subprocess

print("📦 Dò tìm mã nguồn từ dataset...")
codebase_paths = glob.glob('/kaggle/input/**/train_joint.py', recursive=True)
if codebase_paths:
    codebase_dir = os.path.dirname(codebase_paths[0])
    print(f"✅ Found codebase at {codebase_dir}")
    os.system(f"cp -r {codebase_dir}/* /kaggle/working/")
else:
    print("⚠️ Could not find codebase!")
    os.system("unzip -o -q /kaggle/input/*/codebase.zip -d /kaggle/working/")

print("🔗 Linking dataset using glob search...")
os.system("mkdir -p /kaggle/working/data")

# Tìm động dataset
kaggle_train_paths = glob.glob('/kaggle/input/**/train', recursive=True)
if kaggle_train_paths:
    train_dir = kaggle_train_paths[0]
    os.system(f"ln -sfn {train_dir} /kaggle/working/data/train")
    print(f"✅ Linked {train_dir} -> data/train")

kaggle_test_paths = glob.glob('/kaggle/input/**/public_test', recursive=True) or glob.glob('/kaggle/input/**/*test*', recursive=True)
if kaggle_test_paths:
    test_dir = kaggle_test_paths[0]
    os.system(f"ln -sfn {test_dir} /kaggle/working/data/public_test")
    print(f"✅ Linked {test_dir} -> data/public_test")

print("🚀 Khởi chạy quá trình Joint Training...")
os.chdir("/kaggle/working")
# Fix config MODEL_TYPE=sr_vit using inline python script
fix_config_cmd = """
with open('configs/config.py', 'r') as f:
    c = f.read()
c = c.replace('MODEL_TYPE: str = "restran"', 'MODEL_TYPE: str = "sr_vit"')
with open('configs/config.py', 'w') as f:
    f.write(c)
"""
subprocess.run(["python", "-c", fix_config_cmd])

os.system("pip install albumentations")
os.system("python train_joint.py")
PY_EOF

echo "☁️ Đang push Kaggle Kernel..."
yes | .venv/bin/kaggle kernels push -p .

echo "🎉 Xong! Kernel huấn luyện Joint SR-ViT v11 đã được đẩy lên Kaggle."
echo "🔗 Link: https://www.kaggle.com/dotrongminhquan/lrlpr-flow-3-joint-vit-v11"
