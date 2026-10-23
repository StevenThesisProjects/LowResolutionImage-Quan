"""Evaluation script for calculating detailed metrics (Accuracy, CER, F1-Score, Confidence Gap) on validation dataset."""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.models.sr_restran import MultiFrameSRResTran
from src.models.sr_vit import MultiFrameSRViT
from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LPR Model Performance Metrics")
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="sr_restran",
        choices=["crnn", "restran", "end2end", "sr_restran", "sr_vit"],
        help="Default model architecture if cannot auto-detect",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained primary model checkpoint (.pth)",
    )
    parser.add_argument(
        "--ensemble-weights",
        type=str,
        nargs="*",
        default=[],
        help="Optional list of additional checkpoint paths (.pth) for Multi-Model / Multi-Flow Log-Probability Ensembling",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/train",
        help="Path to training/validation dataset root directory",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/evaluation_metrics.json",
        help="Path to output metrics JSON file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--no-stn",
        action="store_true",
        help="Disable STN module",
    )
    parser.add_argument(
        "--no-format-prior",
        action="store_true",
        help="Disable 7-character plate format grammar constraint in decoding",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable contrast-aware test-time augmentation (TTA) in evaluation",
    )
    return parser.parse_args()


def load_any_lpr_model(
    ckpt_path: str,
    config: Config,
    device: torch.device,
    default_arch: str = "sr_restran",
) -> Tuple[torch.nn.Module, str]:
    """Dynamically detects architecture and loads weights for any checkpoint."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    keys = list(state_dict.keys())
    layer_indices = set()
    for k in keys:
        if "layers." in k:
            parts = k.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_indices.add(int(parts[i + 1]))
    detected_tf_layers = max(layer_indices) + 1 if layer_indices else None

    # Architecture heuristics based on state_dict keys
    if any(k.startswith("conv_stem.") or "cross_attn" in k or k == "pos_embed" for k in keys):
        arch = "sr_vit"
        vit_layers = detected_tf_layers if detected_tf_layers is not None else config.SR_VIT_NUM_LAYERS
        model = MultiFrameSRViT(
            num_classes=config.NUM_CLASSES,
            sr_num_blocks=8,
            patch_h=config.SR_VIT_PATCH_H,
            patch_w=config.SR_VIT_PATCH_W,
            embed_dim=config.SR_VIT_EMBED_DIM,
            num_heads=config.SR_VIT_NUM_HEADS,
            num_layers=vit_layers,
            ff_dim=config.SR_VIT_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            max_seq_len=128,
            use_stn=config.USE_STN,
            pretrained=False,
        )
    elif any(k.startswith("sr_net.") for k in keys) and any(k.startswith("ocr_net.") for k in keys):
        arch = "sr_restran"
        tf_layers = detected_tf_layers if detected_tf_layers is not None else config.TRANSFORMER_LAYERS
        model = MultiFrameSRResTran(
            num_classes=config.NUM_CLASSES,
            sr_num_blocks=8,
            sr_scale_factor=2,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=tf_layers,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            pretrained=False,
        )
    elif any(k.startswith("backbone.") for k in keys) and any(k.startswith("transformer.") for k in keys):
        arch = "restran"
        tf_layers = detected_tf_layers if detected_tf_layers is not None else config.TRANSFORMER_LAYERS
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=tf_layers,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            pretrained=False,
        )
    else:
        arch = default_arch
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical_missing = [
        k for k in missing
        if k not in ["sr_net.res_scale_gate", "ocr_net.stn.identity_theta", "stn.identity_theta"]
    ]
    if critical_missing:
        print(f"⚠️ [{os.path.basename(ckpt_path)}] Missing {len(critical_missing)} keys: {critical_missing[:3]}...")
    else:
        matched = len(state_dict) - len(unexpected)
        print(f"✅ [{os.path.basename(ckpt_path)}] Loaded as '{arch}' ({matched} keys matched, {detected_tf_layers or 'N/A'} layers).")

    model = model.to(device)
    model.eval()
    return model, arch


def model_forward_with_tta(
    model: torch.nn.Module,
    arch: str,
    images: torch.Tensor,
    use_tta: bool = True,
) -> torch.Tensor:
    """Runs model forward pass with optional 3-view Contrast TTA and resolution adaptation."""
    B, F_cnt, C, H, W = images.shape
    if arch == "restran" and (H == 16 and W == 64):
        in_imgs = F.interpolate(
            images.view(-1, C, H, W),
            size=(32, 128),
            mode="bilinear",
            align_corners=False,
        ).view(B, F_cnt, C, 32, 128)
    else:
        in_imgs = images

    if not use_tta:
        return model(in_imgs)

    # 3-view Contrast TTA: 1.0x (0.60), 1.08x (0.25), 0.92x (0.15)
    preds_orig = model(in_imgs)
    preds_sharp = model(torch.clamp(in_imgs * 1.08, -1.0, 1.0))
    preds_soft = model(torch.clamp(in_imgs * 0.92, -1.0, 1.0))
    return torch.logsumexp(
        torch.stack(
            [
                preds_orig + math.log(0.60),
                preds_sharp + math.log(0.25),
                preds_soft + math.log(0.15),
            ],
            dim=0,
        ),
        dim=0,
    )


def main():
    args = parse_args()
    config = Config()
    seed_everything(config.SEED)

    device = config.DEVICE
    print(f"🚀 Starting Comprehensive Model Evaluation on device: {device}")

    if args.no_stn:
        config.USE_STN = False

    # 1. Load primary model
    weights_path = args.weights
    print(f"📦 Loading primary weights from: {weights_path}")
    primary_model, primary_arch = load_any_lpr_model(
        ckpt_path=weights_path,
        config=config,
        device=device,
        default_arch=args.model,
    )

    # 2. Load auxiliary ensemble models if requested
    ensemble_models: List[Tuple[torch.nn.Module, str]] = []
    if args.ensemble_weights:
        print(f"🤝 Initializing Multi-Model Ensemble with {len(args.ensemble_weights)} auxiliary checkpoints:")
        for aux_path in args.ensemble_weights:
            aux_m, aux_arch = load_any_lpr_model(
                ckpt_path=aux_path,
                config=config,
                device=device,
                default_arch=args.model,
            )
            ensemble_models.append((aux_m, aux_arch))

    # 3. Determine input resolution and load validation dataset
    has_sr_pipeline = (
        primary_arch in ["sr_restran", "sr_vit", "end2end"]
        or any(a in ["sr_restran", "sr_vit", "end2end"] for _, a in ensemble_models)
    )
    in_h = 16 if has_sr_pipeline else config.IMG_HEIGHT
    in_w = 64 if has_sr_pipeline else config.IMG_WIDTH

    val_ds = MultiFrameDataset(
        root_dir=args.data_dir,
        mode="val",
        split_ratio=config.SPLIT_RATIO,
        img_height=in_h,
        img_width=in_w,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
    )

    if len(val_ds) == 0:
        print(f"❌ ERROR: Validation dataset is empty in {args.data_dir}")
        sys.exit(1)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    print(f"📊 Loaded {len(val_ds)} validation tracks ({len(val_loader)} batches, input size: {in_h}x{in_w})")

    # 4. Evaluation Loop
    total_tracks = 0
    exact_matches = 0
    total_edit_distance = 0
    total_characters = 0

    char_tp = defaultdict(int)
    char_fp = defaultdict(int)
    char_fn = defaultdict(int)

    char_confusion = Counter()
    length_acc = defaultdict(lambda: {"correct": 0, "total": 0})

    confidences = []
    correct_flags = []

    print("🔮 Evaluating Validation Set...")
    with torch.no_grad():
        for images, targets, target_lengths, labels_text, track_ids in tqdm(val_loader, desc="Evaluation"):
            images = images.to(device)

            preds_primary = model_forward_with_tta(
                primary_model,
                primary_arch,
                images,
                use_tta=(not args.no_tta),
            )

            if ensemble_models:
                aux_preds_list = [
                    model_forward_with_tta(m, a, images, use_tta=(not args.no_tta))
                    for m, a in ensemble_models
                ]
                w_primary = 0.35
                w_aux = (1.0 - w_primary) / len(aux_preds_list)
                all_log_probs = [preds_primary + math.log(w_primary)] + [
                    lp_aux + math.log(w_aux) for lp_aux in aux_preds_list
                ]
                preds = torch.logsumexp(torch.stack(all_log_probs, dim=0), dim=0)
            else:
                preds = preds_primary

            decoded_list = decode_with_confidence(
                preds,
                config.IDX2CHAR,
                use_format_prior=(not args.no_format_prior),
            )

            for i, (pred_text, conf) in enumerate(decoded_list):
                gt_text = labels_text[i]
                total_tracks += 1

                confidences.append(conf)
                is_correct = (pred_text == gt_text)
                correct_flags.append(is_correct)

                if is_correct:
                    exact_matches += 1

                # Edit Distance & CER
                dist = levenshtein_distance(pred_text, gt_text)
                total_edit_distance += dist
                total_characters += len(gt_text)

                # Length breakdown
                gt_len = len(gt_text)
                length_acc[gt_len]["total"] += 1
                if is_correct:
                    length_acc[gt_len]["correct"] += 1

                # Character level precision/recall metrics
                pred_chars = Counter(pred_text)
                gt_chars = Counter(gt_text)

                all_unique_chars = set(pred_chars.keys()).union(set(gt_chars.keys()))
                for char in all_unique_chars:
                    tp = min(pred_chars[char], gt_chars[char])
                    fp = max(0, pred_chars[char] - gt_chars[char])
                    fn = max(0, gt_chars[char] - pred_chars[char])

                    char_tp[char] += tp
                    char_fp[char] += fp
                    char_fn[char] += fn

                # Misclassified character pairs (if lengths match)
                if not is_correct and len(pred_text) == len(gt_text):
                    for p_c, g_c in zip(pred_text, gt_text):
                        if p_c != g_c:
                            char_confusion[(g_c, p_c)] += 1

    # 5. Compute Aggregate Metrics
    exact_match_acc = (exact_matches / total_tracks) * 100 if total_tracks > 0 else 0.0
    cer = (total_edit_distance / total_characters) * 100 if total_characters > 0 else 0.0
    normalized_accuracy = max(0.0, 100.0 - cer)

    # Confidence Gap (Slide 10 ICPR 2026 Official Ranking Metric)
    correct_confs = [c for c, is_corr in zip(confidences, correct_flags) if is_corr]
    incorrect_confs = [c for c, is_corr in zip(confidences, correct_flags) if not is_corr]
    mean_correct_conf = float(np.mean(correct_confs)) if correct_confs else 0.0
    mean_incorrect_conf = float(np.mean(incorrect_confs)) if incorrect_confs else 0.0
    confidence_gap = float(mean_correct_conf - mean_incorrect_conf)

    # Character-level Macro Precision, Recall, F1
    precisions = []
    recalls = []
    f1s = []

    all_chars = sorted(list(set(char_tp.keys()).union(set(char_fp.keys())).union(set(char_fn.keys()))))
    for char in all_chars:
        tp = char_tp[char]
        fp = char_fp[char]
        fn = char_fn[char]

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    total_tp = sum(char_tp.values())
    total_fp = sum(char_fp.values())
    total_fn = sum(char_fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Format JSON Output
    metrics_summary = {
        "model_type": primary_arch,
        "weights_file": weights_path,
        "ensemble_weights": args.ensemble_weights,
        "total_val_tracks": total_tracks,
        "exact_match_accuracy": round(exact_match_acc, 2),
        "confidence_gap": round(confidence_gap, 4),
        "mean_correct_confidence": round(mean_correct_conf, 4),
        "mean_incorrect_confidence": round(mean_incorrect_conf, 4),
        "character_error_rate_cer": round(cer, 2),
        "normalized_edit_accuracy": round(normalized_accuracy, 2),
        "macro_metrics": {
            "precision": round(macro_precision * 100, 2),
            "recall": round(macro_recall * 100, 2),
            "f1_score": round(macro_f1 * 100, 2)
        },
        "micro_metrics": {
            "precision": round(micro_p * 100, 2),
            "recall": round(micro_r * 100, 2),
            "f1_score": round(micro_f1 * 100, 2)
        },
        "top_10_character_confusions": [
            {"gt": gt, "pred": pred, "count": count}
            for (gt, pred), count in char_confusion.most_common(10)
        ]
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # 6. Print Evaluation Report
    print("\n" + "=" * 65)
    title_suffix = f" + ENSEMBLE ({len(ensemble_models)} models)" if ensemble_models else " SINGLE MODEL"
    print(f"📊 EXPLICIT EVALUATION REPORT: {primary_arch.upper()}{title_suffix}")
    print("=" * 65)
    print(f"🎯 Exact Match Accuracy (Sequence Acc):  {exact_match_acc:.2f}% ({exact_matches}/{total_tracks})")
    print(f"📈 Confidence Gap (Slide 10):            {confidence_gap:+.4f} (Correct: {mean_correct_conf:.4f} vs Incorrect: {mean_incorrect_conf:.4f})")
    print(f"🔤 Character Error Rate (CER):           {cer:.2f}%")
    print(f"📏 Normalized Edit Distance Accuracy:    {normalized_accuracy:.2f}%")
    print("-" * 65)
    print(f"✨ Macro F1-Score:                        {macro_f1 * 100:.2f}%")
    print(f"✨ Macro Precision:                       {macro_precision * 100:.2f}%")
    print(f"✨ Macro Recall:                          {macro_recall * 100:.2f}%")
    print("-" * 65)
    print(f"⚡ Micro F1-Score:                        {micro_f1 * 100:.2f}%")
    print(f"⚡ Micro Precision:                       {micro_p * 100:.2f}%")
    print(f"⚡ Micro Recall:                          {micro_r * 100:.2f}%")
    print("=" * 65)

    print("\n📌 ACCURACY BY PLATE LENGTH (Length Breakdown):")
    for length in sorted(length_acc.keys()):
        stats = length_acc[length]
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
        print(f"  • Length {length} characters: {acc:.2f}% ({stats['correct']}/{stats['total']} tracks)")

    if char_confusion:
        print("\n⚠️ TOP 5 MOST COMMON CHARACTER CONFUSIONS (GroundTruth -> Predicted):")
        for (gt, pred), count in char_confusion.most_common(5):
            print(f"  • '{gt}' misrecognized as '{pred}': {count} times")
    print("=" * 65)
    print(f"💾 Full evaluation summary saved to: {args.output_json}\n")


if __name__ == "__main__":
    main()
