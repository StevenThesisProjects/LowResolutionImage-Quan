import json
import os

files_to_embed = [
    "configs/config.py",
    "src/data/transforms.py",
    "src/data/dataset.py",
    "src/models/components.py",
    "src/models/crnn.py",
    "src/models/restran.py",
    "src/utils/common.py",
    "src/utils/postprocess.py",
    "src/training/trainer.py",
    "train.py"
]

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# MultiFrame-LPR (ThesisLRLPR-SR-Transformer): Kaggle GPU Training\n",
            "\n",
            "This notebook runs PyTorch training using NVIDIA GPU CUDA acceleration on Kaggle."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "print('PyTorch Version:', torch.__version__)\n",
            "print('CUDA Available:', torch.cuda.is_available())\n",
            "if not torch.cuda.is_available():\n",
            "    raise RuntimeError(\n",
            "        '❌ ERROR: CUDA GPU is not enabled for this Kaggle Notebook session!\\n' \n",
            "        'Please enable GPU T4 x2 or P100 in the right-hand panel -> Accelerator -> GPU T4 x2, then Save & Run All.'\n",
            "    )\n",
            "print('🚀 CUDA GPU detected successfully:', torch.cuda.get_device_name(0))\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "!git clone https://github.com/QuanDo610/ThesisLRLPR-SR-Transformer.git repo || true\n",
            "if os.path.exists('repo'):\n",
            "    !cp -rf repo/* ./\n",
            "    print('✅ Copied repository files to current working directory')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "os.makedirs('configs', exist_ok=True)\n",
            "os.makedirs('src/data', exist_ok=True)\n",
            "os.makedirs('src/models', exist_ok=True)\n",
            "os.makedirs('src/utils', exist_ok=True)\n",
            "os.makedirs('src/training', exist_ok=True)\n",
            "with open('src/__init__.py', 'w') as f: pass\n",
            "with open('src/data/__init__.py', 'w') as f: pass\n",
            "with open('src/models/__init__.py', 'w') as f: pass\n",
            "with open('src/utils/__init__.py', 'w') as f: pass\n",
            "with open('src/training/__init__.py', 'w') as f: pass\n",
            "with open('configs/__init__.py', 'w') as f: pass\n"
        ]
    }
]

# Add each source file as a %%writefile cell
for filepath in files_to_embed:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        
        lines = [f"%%writefile {filepath}\n"] + [line + "\n" for line in content.splitlines()]
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

# Add dataset linking cell
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import glob\n",
        "import os\n",
        "\n",
        "kaggle_train_paths = glob.glob('/kaggle/input/**/train', recursive=True)\n",
        "if kaggle_train_paths:\n",
        "    train_dir = kaggle_train_paths[0]\n",
        "    os.makedirs('data', exist_ok=True)\n",
        "    !ln -sfn {train_dir} data/train\n",
        "    print(f\"✅ Linked dataset: {train_dir} -> data/train\")\n",
        "else:\n",
        "    print(\"⚠️ Could not find train directory in /kaggle/input/\")\n",
        "\n",
        "kaggle_test_paths = glob.glob('/kaggle/input/**/public_test', recursive=True) or glob.glob('/kaggle/input/**/*test*', recursive=True)\n",
        "if kaggle_test_paths:\n",
        "    test_dir = kaggle_test_paths[0]\n",
        "    os.makedirs('data', exist_ok=True)\n",
        "    !ln -sfn {test_dir} data/public_test\n",
        "    print(f\"✅ Linked test dataset: {test_dir} -> data/public_test\")\n"
    ]
})

# Add requirements installation cell
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!pip install albumentations opencv-python-headless\n"
    ]
})

# Add execution cell for training ResTranOCR
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Train ResTranOCR Model on Kaggle GPU CUDA\n",
        "!python train.py --model restran --experiment-name restran_kaggle_gpu --epochs 30 --batch-size 64 --require-gpu\n"
    ]
})

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs('kaggle_kernel', exist_ok=True)
with open('kaggle_kernel/kaggle_notebook.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

metadata = {
  "id": "dotrongminhquan/multiframe-lpr-train",
  "title": "MultiFrame LPR Train",
  "code_file": "kaggle_notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": True,
  "dataset_sources": [
    "trunghiu/icpr-car-plate-dataset",
    "nhttinon/icpr-public-test"
  ],
  "competition_sources": [],
  "kernel_sources": []
}

with open('kaggle_kernel/kernel-metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Built self-contained kaggle_notebook.ipynb and kernel-metadata.json successfully!")
