#!/bin/bash
# ==============================================================================
# 🎯 MISSION 80.5+: DEEP PUSH FOR FLOW 2 (RESTRAN) TOWARDS >= 80.50% & >= 83.50%
# Local execution on Apple Silicon Mac (MPS)
# Starting from best checkpoint: results/decoupled_restran_best.pth (75.38%)
# Targeting the 128 "Off-By-One" error tracks with:
# - Expanded hard character curriculum sampling (8,192 tracks/epoch, 2x increase)
# - Stronger angular margin regularizer (--lambda-margin 4.0) for H-N, W-M, Y-T, 8-6, V-Y
# - Rule 10 hard time guard (--max-hours 1.1)
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

PYTHON="$DIR/.venv/bin/python"
LOG_FILE="$DIR/results/run_push_80_plus.log"
mkdir -p "$DIR/results"

echo "========================================================================" | tee -a "$LOG_FILE"
echo "🚀 MISSION 80.5+: TARGETED JOINT TRAINING FOR THESIS DEFENSE (>=80.50%)" | tee -a "$LOG_FILE"
echo "   Date & Time   : $(date)" | tee -a "$LOG_FILE"
echo "   Device Target : Apple Silicon MPS (Local)" | tee -a "$LOG_FILE"
echo "   Init Weight   : results/decoupled_restran_best.pth (75.38%)" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"

# Backup 75.38% baseline checkpoint
if [ -f "results/decoupled_restran_best.pth" ]; then
    cp -v results/decoupled_restran_best.pth results/decoupled_restran_25ep_75.38.pth | tee -a "$LOG_FILE"
fi

$PYTHON train_decoupled.py \
    --model restran \
    --frozen-epochs 0 \
    --joint-epochs 15 \
    --samples-per-epoch 8192 \
    --batch-size 48 \
    --lr-sr-stage2 3e-6 \
    --lr-ocr-stage2 3.5e-5 \
    --lambda-sr 0.1 \
    --lambda-margin 4.0 \
    --max-hours 1.1 \
    --init-weights results/decoupled_restran_best.pth \
    --output-dir results 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "🔍 Running Comprehensive Final Evaluation on Scenario-B Validation..." | tee -a "$LOG_FILE"
$PYTHON evaluate.py \
    -m sr_restran \
    --weights results/decoupled_restran_best.pth \
    --output-json results/evaluation_metrics_sr_restran.json 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"
echo "🎉 MISSION 80.5+ TRAINING COMPLETE!" | tee -a "$LOG_FILE"
echo "   Date & Time: $(date)" | tee -a "$LOG_FILE"
echo "========================================================================" | tee -a "$LOG_FILE"
