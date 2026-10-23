#!/bin/bash
# ==============================================================================
# 🚀 FULL-DATASET TRAINING ROUND 2: FLOW 2 RESTRAN (TARGET >= 80.50% THESIS GOAL)
# Architecture : MultiFrameSRResTran (Flow 2 Proposed)
# Device       : Apple Silicon MPS (Local Mac)
# Training Set : 100% FULL DATASET (38,002 tracks / epoch)
# Anti-Sleep   : caffeinate -dimsu enabled to prevent macOS idle sleep
# Safeguards   : Rule 10 hard time guard (--max-hours 5.5) + Auto Backup 76.68%
# Checkpoint   : Warm-start from best 76.68% (backed up before run)
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

PYTHON="$DIR/.venv/bin/python"
LOG_FILE="$DIR/results/run_full_train_round2.log"
mkdir -p "$DIR/results"

echo "========================================================================" | tee -a "$LOG_FILE"
echo "🚀 STARTING FULL-DATASET TRAINING ROUND 2 (100% FULL DATASET)" | tee -a "$LOG_FILE"
echo "   Date & Time       : $(date)" | tee -a "$LOG_FILE"
echo "   Device Target     : Apple Silicon MPS (Local Mac)" | tee -a "$LOG_FILE"
echo "   Base Weights      : results/decoupled_restran_best.pth (Acc: 76.68%, CER: 6.49%)" | tee -a "$LOG_FILE"
echo "   Samples / Epoch   : 38,002 tracks (100% FULL DATASET)" | tee -a "$LOG_FILE"
echo "   Batch Size        : 48 (792 batches / epoch)" | tee -a "$LOG_FILE"
echo "   Epochs Planned    : 15 Epochs Full Dataset" | tee -a "$LOG_FILE"
echo "   Est. Time / Epoch : ~18.5 minutes" | tee -a "$LOG_FILE"
echo "   Est. Total Time   : ~4.6 hours" | tee -a "$LOG_FILE"
echo "   Hard Time Cap     : 5.5 hours (--max-hours 5.5)" | tee -a "$LOG_FILE"
echo "   LR Configuration  : OCR LR=1.5e-5 (fine-tune mode), SR LR=2.0e-6" | tee -a "$LOG_FILE"
echo "   Loss Setup        : HardCharacterCTCLoss (lambda_margin=4.0) + SRLoss (0.1)" | tee -a "$LOG_FILE"
echo "   Power Assertion   : caffeinate -dimsu (macOS Sleep Prevention)" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"

# 1. Safeguard: Backup current best 76.68% checkpoint
if [ -f "results/decoupled_restran_best.pth" ]; then
    echo "📦 Backing up 76.68% checkpoint..." | tee -a "$LOG_FILE"
    cp -v results/decoupled_restran_best.pth results/decoupled_restran_backup_76.68.pth | tee -a "$LOG_FILE"
fi

# 2. Execute Full Dataset Training with caffeinate
echo "🔥 Launching Training Loop..." | tee -a "$LOG_FILE"
/usr/bin/caffeinate -dimsu $PYTHON train_decoupled.py \
    --model restran \
    --frozen-epochs 0 \
    --joint-epochs 15 \
    --samples-per-epoch 0 \
    --batch-size 48 \
    --lr-sr-stage2 2e-6 \
    --lr-ocr-stage2 1.5e-5 \
    --lambda-sr 0.1 \
    --lambda-margin 4.0 \
    --max-hours 5.5 \
    --init-weights results/decoupled_restran_best.pth \
    --output-dir results 2>&1 | tee -a "$LOG_FILE"

# 3. Final Comprehensive Validation
echo "" | tee -a "$LOG_FILE"
echo "🔍 Running Comprehensive Final Evaluation on Scenario-B Validation..." | tee -a "$LOG_FILE"
$PYTHON evaluate.py \
    -m sr_restran \
    --weights results/decoupled_restran_best.pth \
    --output-json results/evaluation_metrics_sr_restran.json 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"
echo "🎉 TRAINING ROUND 2 COMPLETED SUCCESSFULLY!" | tee -a "$LOG_FILE"
echo "   Date & Time: $(date)" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"
