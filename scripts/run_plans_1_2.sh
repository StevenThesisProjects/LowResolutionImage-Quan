#!/bin/bash
# ==============================================================================
# 🎯 AUTOMATED EXECUTION: PHƯƠNG ÁN 1 & PHƯƠNG ÁN 2
# Job 1: Flow 2 (sr_restran) Joint Deep Fine-Tuning (10 epochs, Stage 2 only)
# Job 2: Flow 3 (sr_vit) Two-Stage Decoupled Training (5 Frozen + 5 Joint = 10 epochs)
# Guaranteed local execution on Mac MPS with Rule 10 time cap.
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

PYTHON="$DIR/.venv/bin/python"
LOG_FILE="$DIR/results/run_plans_1_2.log"
mkdir -p "$DIR/results"

echo "========================================================================" | tee -a "$LOG_FILE"
echo "🚀 STARTING DUAL-PLAN TRAINING PIPELINE" | tee -a "$LOG_FILE"
echo "   Date & Time: $(date)" | tee -a "$LOG_FILE"
echo "   Working Dir: $DIR" | tee -a "$LOG_FILE"
echo "   Python Exec: $PYTHON" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"

# Backup baseline checkpoint
if [ -f "results/decoupled_restran_best.pth" ]; then
    cp -v results/decoupled_restran_best.pth results/decoupled_restran_15ep_74.27.pth | tee -a "$LOG_FILE"
fi

# ==============================================================================
# 🔷 JOB 1: PHƯƠNG ÁN 1 - FLOW 2 DEEP JOINT FINE-TUNING
# ==============================================================================
echo "" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"
echo "▶️ [1/2] RUNNING PLAN 1: Flow 2 Joint Deep Fine-Tuning (Target: 78-80%)" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"

$PYTHON train_decoupled.py \
    --model restran \
    --frozen-epochs 0 \
    --joint-epochs 10 \
    --samples-per-epoch 4096 \
    --batch-size 32 \
    --lr-sr-stage2 5e-6 \
    --lr-ocr-stage2 2.5e-5 \
    --lambda-sr 0.1 \
    --max-hours 0.4 \
    --init-weights results/decoupled_restran_best.pth \
    --output-dir results 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "🔍 Running Comprehensive Evaluation for Flow 2 (sr_restran)..." | tee -a "$LOG_FILE"
$PYTHON evaluate.py \
    -m sr_restran \
    --weights results/decoupled_restran_best.pth \
    --output-json results/evaluation_metrics_sr_restran.json 2>&1 | tee -a "$LOG_FILE"

# ==============================================================================
# 🔶 JOB 2: PHƯƠNG ÁN 2 - FLOW 3 SR-VIT TWO-STAGE DECOUPLED TRAINING
# ==============================================================================
echo "" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"
echo "▶️ [2/2] RUNNING PLAN 2: Flow 3 SR-ViT Decoupled Training (5 Frozen + 5 Joint)" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"

$PYTHON train_decoupled.py \
    --model sr_vit \
    --frozen-epochs 5 \
    --joint-epochs 5 \
    --samples-per-epoch 4096 \
    --batch-size 32 \
    --lr-ocr-stage1 1.5e-4 \
    --lr-ocr-stage2 5e-5 \
    --lr-sr-stage2 1e-5 \
    --lambda-sr 0.1 \
    --max-hours 0.4 \
    --init-weights results/flow3_warmstart_init.pth \
    --output-dir results 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "🔍 Running Comprehensive Evaluation for Flow 3 (sr_vit)..." | tee -a "$LOG_FILE"
$PYTHON evaluate.py \
    -m sr_vit \
    --weights results/decoupled_sr_vit_best.pth \
    --output-json results/evaluation_metrics_sr_vit.json 2>&1 | tee -a "$LOG_FILE"

# ==============================================================================
# 🏁 COMPLETION SUMMARY
# ==============================================================================
echo "" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"
echo "🎉 ALL JOBS COMPLETED SUCCESSFULLY!" | tee -a "$LOG_FILE"
echo "   Date & Time: $(date)" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"
