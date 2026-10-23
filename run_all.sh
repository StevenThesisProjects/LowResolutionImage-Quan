#!/bin/bash
echo "=== 1. Starting 40 Epochs End-to-End Joint Training ==="
PYTHONPATH=.:.venv/lib/python3.11/site-packages /Users/admin/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11 train_joint.py

echo "=== 2. Running Comprehensive Validation on Best Checkpoint ==="
PYTHONPATH=.:.venv/lib/python3.11/site-packages /Users/admin/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11 evaluate_end2end_val.py > tests/eval_end2end_val_40epochs.log

echo "=== 3. Running Benchmark & Ablation Evaluation ==="
PYTHONPATH=.:.venv/lib/python3.11/site-packages /Users/admin/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11 scripts/evaluate_ablation.py > tests/eval_ablation_final.log

echo "=== 4. Running Confusion Matrix Analysis ==="
PYTHONPATH=.:.venv/lib/python3.11/site-packages /Users/admin/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11 scripts/eval_confusion.py > tests/eval_confusion_final.log

echo "=== 🎯 ALL 40 EPOCHS TRAINING AND EVALUATIONS COMPLETED ==="
