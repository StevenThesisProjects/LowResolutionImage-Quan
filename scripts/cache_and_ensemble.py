"""Fair scientific evaluation of all trained models and uniform (1/M) 7-slot aligned ensembles across 4 ablation modes."""
import json
import math
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from evaluate import levenshtein_distance, load_any_lpr_model, model_forward_with_tta
from src.data.dataset import MultiFrameDataset
from src.utils.postprocess import (
    decode_with_confidence,
    extract_7slot_ctc_tensor,
    slots_to_canonical_ctc,
)


def evaluate_predictions(
    preds: torch.Tensor,
    labels: List[str],
    config: Config,
    use_format_prior: bool = False,
) -> Tuple[float, float, float, int, List[Tuple[str, float]]]:
    """Decodes a [B, T, C] CTC log-probability tensor and computes metrics."""
    decoded_list = decode_with_confidence(preds, config.IDX2CHAR, use_format_prior=use_format_prior)

    exact_matches = 0
    total_edit_distance = 0
    total_chars = 0
    confidences = []
    correct_flags = []

    for (pred_text, conf), gt_text in zip(decoded_list, labels):
        is_corr = (pred_text == gt_text)
        confidences.append(conf)
        correct_flags.append(is_corr)
        if is_corr:
            exact_matches += 1
        dist = levenshtein_distance(pred_text, gt_text)
        total_edit_distance += dist
        total_chars += len(gt_text)

    exact_match_acc = (exact_matches / len(labels)) * 100.0 if labels else 0.0
    cer = (total_edit_distance / total_chars) * 100.0 if total_chars > 0 else 0.0

    corr_c = [c for c, ok in zip(confidences, correct_flags) if ok]
    incorr_c = [c for c, ok in zip(confidences, correct_flags) if not ok]
    gap = (np.mean(corr_c) - np.mean(incorr_c)) if (corr_c and incorr_c) else 0.0

    return exact_match_acc, cer, float(gap), exact_matches, decoded_list


def uniform_7slot_ensemble(
    preds_list: List[torch.Tensor],
    anchor_preds: torch.Tensor,
) -> torch.Tensor:
    """Combines M models with strictly uniform 1/M weights in 7-slot aligned CTC space."""
    slot_tensors = []
    valid_masks = []
    for raw_p in preds_list:
        s_lp, v_m = extract_7slot_ctc_tensor(raw_p, anchor_preds=anchor_preds, temperature=1.0)
        slot_tensors.append(s_lp)
        valid_masks.append(v_m)

    B, _, C = slot_tensors[0].shape
    num_m = len(slot_tensors)
    uniform_w = 1.0 / num_m

    ens_slots = torch.zeros((B, 7, C), dtype=torch.float32)
    for b_i in range(B):
        active_lps = [s_t[b_i] for s_t, v_m in zip(slot_tensors, valid_masks) if v_m[b_i]]
        if active_lps:
            w_each = 1.0 / len(active_lps)
            stacked = torch.stack([lp + math.log(w_each) for lp in active_lps], dim=0)
            ens_slots[b_i] = torch.logsumexp(stacked, dim=0)
        else:
            ens_slots[b_i] = slot_tensors[0][b_i]

    return slots_to_canonical_ctc(ens_slots)


def main():
    config = Config()
    device = config.DEVICE

    data_root = "dataset/data/train" if os.path.exists("dataset/data/train") else "data/train"
    val_ds_16 = MultiFrameDataset(
        root_dir=data_root,
        mode="val",
        split_ratio=config.SPLIT_RATIO,
        img_height=16,
        img_width=64,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
    )
    val_loader_16 = DataLoader(
        val_ds_16,
        batch_size=64,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    val_ds_32 = MultiFrameDataset(
        root_dir=data_root,
        mode="val",
        split_ratio=config.SPLIT_RATIO,
        img_height=32,
        img_width=128,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
    )
    val_loader_32 = DataLoader(
        val_ds_32,
        batch_size=64,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    models_info = [
        ("flow1_crnn", "results/crnn_baseline_full_best.pth"),
        ("flow1_restran", "results/restran_kaggle_gpu_best.pth"),
        ("decoupled", "results/decoupled_restran_best.pth"),
        ("flow2_v21", "results/sr_restran_flow2_plus10_best.pth"),
        ("flow2_kaggle", "results/sr_restran_flow2_kaggle_plus10_best.pth"),
        ("flow2_v22", "results/sr_restran_flow2_v22_best.pth"),
        ("flow2_swa", "results/sr_restran_flow2_swa_best.pth"),
        ("flow3_vit", "results/sr_vit_flow3_best.pth"),
        ("flow3_v22", "results/sr_vit_flow3_v22_best.pth"),
        ("flow3_swa", "results/sr_vit_flow3_swa_best.pth"),
    ]

    existing_models = [(name, path) for name, path in models_info if os.path.exists(path)]
    print(f"📦 Found {len(existing_models)} checkpoints for scientific ablation evaluation:")
    for name, path in existing_models:
        print(f"  • {name:15s}: {path}")

    loaded_models = []
    for name, path in existing_models:
        m, arch = load_any_lpr_model(path, config, device)
        loaded_models.append((name, m, arch))

    preds_no_tta: Dict[str, List[torch.Tensor]] = {name: [] for name, _, _ in loaded_models}
    preds_tta: Dict[str, List[torch.Tensor]] = {name: [] for name, _, _ in loaded_models}
    all_labels: List[str] = []

    print("\n🔮 Running single-view (No TTA) and 4-view Photometric TTA inference...")
    iter_32 = iter(val_loader_32)
    with torch.no_grad():
        for batch_16 in tqdm(val_loader_16, desc="Evaluating"):
            images_16, _, _, labels_text, _ = batch_16
            images_16 = images_16.to(device)
            images_32 = next(iter_32)[0].to(device)
            all_labels.extend(labels_text)

            for name, m, arch in loaded_models:
                p_single = model_forward_with_tta(m, arch, images_16, use_tta=False, images_32=images_32)
                p_tta = model_forward_with_tta(m, arch, images_16, use_tta=True, images_32=images_32)
                preds_no_tta[name].append(p_single.cpu())
                preds_tta[name].append(p_tta.cpu())

    raw_no_tta = {name: torch.cat(preds_no_tta[name], dim=0) for name in preds_no_tta}
    raw_tta = {name: torch.cat(preds_tta[name], dim=0) for name in preds_tta}

    ablation_report = {"models": {}, "uniform_ensembles": {}}

    print("\n" + "=" * 112)
    print(f"{'Model / Paradigm':20s} | {'Pure Greedy (No TTA)':21s} | {'Pure Greedy (+TTA)':21s} | {'+Uniform Syntax (NoTTA)':21s} | {'+Uniform Syntax (+TTA)':21s}")
    print("=" * 112)

    for name in raw_no_tta:
        a1, cer1, g1, c1, _ = evaluate_predictions(raw_no_tta[name], all_labels, config, use_format_prior=False)
        a2, cer2, g2, c2, _ = evaluate_predictions(raw_tta[name], all_labels, config, use_format_prior=False)
        a3, cer3, g3, c3, _ = evaluate_predictions(raw_no_tta[name], all_labels, config, use_format_prior=True)
        a4, cer4, g4, c4, _ = evaluate_predictions(raw_tta[name], all_labels, config, use_format_prior=True)

        print(
            f"{name:20s} | {a1:6.2f}% ({c1:3d}, {cer1:4.2f}%) "
            f"| {a2:6.2f}% ({c2:3d}, {cer2:4.2f}%) "
            f"| {a3:6.2f}% ({c3:3d}, {cer3:4.2f}%) "
            f"| {a4:6.2f}% ({c4:3d}, {cer4:4.2f}%)"
        )
        ablation_report["models"][name] = {
            "pure_greedy_no_tta": {"acc": round(a1, 2), "matches": c1, "cer": round(cer1, 2), "gap": round(g1, 4)},
            "pure_greedy_tta": {"acc": round(a2, 2), "matches": c2, "cer": round(cer2, 2), "gap": round(g2, 4)},
            "uniform_syntax_no_tta": {"acc": round(a3, 2), "matches": c3, "cer": round(cer3, 2), "gap": round(g3, 4)},
            "uniform_syntax_tta": {"acc": round(a4, 2), "matches": c4, "cer": round(cer4, 2), "gap": round(g4, 4)},
        }

    # Evaluate Strictly Uniform (1/M) Multi-Paradigm Ensembles (1 representative per paradigm)
    anchor_key = "flow2_swa" if "flow2_swa" in raw_tta else list(raw_tta.keys())[0]
    paradigm_combos = [
        ("Uniform_3Flow (F2+F3+Dec)", ["flow2_swa", "flow3_vit", "decoupled"]),
        ("Uniform_4Paradigm (F1+F2+F3+Dec)", ["flow2_swa", "flow3_vit", "decoupled", "flow1_crnn"]),
        ("Uniform_5Model (F1+F2_swa+F2_v22+F3+Dec)", ["flow2_swa", "flow2_v22", "flow3_vit", "decoupled", "flow1_crnn"]),
    ]

    print("-" * 112)
    for ens_title, keys in paradigm_combos:
        avail_keys = [k for k in keys if k in raw_tta]
        if len(avail_keys) >= 2:
            ens_no_tta = uniform_7slot_ensemble([raw_no_tta[k] for k in avail_keys], raw_no_tta[anchor_key])
            ens_tta = uniform_7slot_ensemble([raw_tta[k] for k in avail_keys], raw_tta[anchor_key])

            a1, cer1, g1, c1, _ = evaluate_predictions(ens_no_tta, all_labels, config, use_format_prior=False)
            a2, cer2, g2, c2, _ = evaluate_predictions(ens_tta, all_labels, config, use_format_prior=False)
            a3, cer3, g3, c3, _ = evaluate_predictions(ens_no_tta, all_labels, config, use_format_prior=True)
            a4, cer4, g4, c4, _ = evaluate_predictions(ens_tta, all_labels, config, use_format_prior=True)

            print(
                f"{ens_title[:20]:20s} | {a1:6.2f}% ({c1:3d}, {cer1:4.2f}%) "
                f"| {a2:6.2f}% ({c2:3d}, {cer2:4.2f}%) "
                f"| {a3:6.2f}% ({c3:3d}, {cer3:4.2f}%) "
                f"| {a4:6.2f}% ({c4:3d}, {cer4:4.2f}%)"
            )
            ablation_report["uniform_ensembles"][ens_title] = {
                "members": avail_keys,
                "weights": [round(1.0 / len(avail_keys), 4)] * len(avail_keys),
                "pure_greedy_no_tta": {"acc": round(a1, 2), "matches": c1, "cer": round(cer1, 2), "gap": round(g1, 4)},
                "pure_greedy_tta": {"acc": round(a2, 2), "matches": c2, "cer": round(cer2, 2), "gap": round(g2, 4)},
                "uniform_syntax_no_tta": {"acc": round(a3, 2), "matches": c3, "cer": round(cer3, 2), "gap": round(g3, 4)},
                "uniform_syntax_tta": {"acc": round(a4, 2), "matches": c4, "cer": round(cer4, 2), "gap": round(g4, 4)},
            }
    print("=" * 112)

    os.makedirs("results", exist_ok=True)
    out_path = "results/eval_scientific_ablation_v25.json"
    with open(out_path, "w") as f:
        json.dump(ablation_report, f, indent=2)
    print(f"\n💾 Saved complete scientific ablation table to {out_path}")


if __name__ == "__main__":
    main()
