import os
import subprocess
import glob

print("📦 Dò tìm mã nguồn từ dataset...")
codebase_paths = glob.glob('/kaggle/input/**/train_decoupled.py', recursive=True)
if codebase_paths:
    codebase_dir = os.path.dirname(codebase_paths[0])
    os.system(f"cp -r {codebase_dir}/* /kaggle/working/")
else:
    os.system("unzip -o -q /kaggle/input/*/codebase.zip -d /kaggle/working/")

os.system("mkdir -p /kaggle/working/data")

kaggle_train_paths = glob.glob('/kaggle/input/**/train', recursive=True)
if kaggle_train_paths:
    train_dir = kaggle_train_paths[0]
    os.system(f"ln -sfn {train_dir} /kaggle/working/data/train")

kaggle_test_paths = glob.glob('/kaggle/input/**/public_test', recursive=True) or glob.glob('/kaggle/input/**/*test*', recursive=True)
if kaggle_test_paths:
    test_dir = kaggle_test_paths[0]
    os.system(f"ln -sfn {test_dir} /kaggle/working/data/public_test")

print("🚀 Khởi chạy quá trình Two-Stage Training (Flow 2 - ResTran)...")
os.chdir("/kaggle/working")
# Đảm bảo cấu hình config.py trỏ về "restran"
fix_config_cmd = """
with open('configs/config.py', 'r') as f:
    c = f.read()
c = c.replace('MODEL_TYPE: str = "sr_vit"', 'MODEL_TYPE: str = "restran"')
with open('configs/config.py', 'w') as f:
    f.write(c)
"""
subprocess.run(["python", "-c", fix_config_cmd])

os.system("pip install albumentations")
os.system("python train_decoupled.py --model restran")
