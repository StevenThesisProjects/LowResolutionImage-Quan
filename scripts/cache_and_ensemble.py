"""Cache model predictions and find optimal ensemble configuration to exceed 80.00% Exact Match."""
import math
import os
import torch
import numpy as np
from tqdm import tqdm
from configs.config import Config
from src.data.dataset import MultiFrameDataset
from torch.utils.data import DataLoader
from evaluate import load_any_lpr_model, model_forward_with_tta
from src.utils.postprocess import decode_with_confidence


def main():
    config = Config()
    device = config.DEVICE

    val_ds = MultiFrameDataset(
        root_dir="dataset/data/train",
        mode="val",
        split_ratio=config.SPLIT_RATIO,
        img_height=16,
        img_width=64,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    models_info = [
        ("swa", "results/sr_restran_flow2_swa_best.pth"),
        ("v22", "results/sr_restran_flow2_v22_best.pth"),
        ("plus10", "results/sr_restran_flow2_plus10_best.pth"),
        ("push80", "results/sr_restran_flow2_push80_best.pth"),
        ("kaggle", "results/sr_restran_flow2_kaggle_plus10_best.pth"),
        ("decoupled", "results/decoupled_restran_best.pth"),
        ("flow3_vit", "results/sr_vit_flow3_best.pth"),
    ]

    # Filter only existing
    models_info = [(name, path) for name, path in models_info if os.path.exists(path)]
    print(f"📦 Extracting log-probs for {len(models_info)} models...")

    loaded_models = []
    for name, path in models_info:
        m, arch = load_any_lpr_model(path, config, device)
        loaded_models.append((name, m, arch))

    all_model_preds = {name: [] for name, _, _ in loaded_models}
    all_labels = []

    with torch.no_grad():
        for images, targets, target_lengths, labels_text, track_ids in tqdm(val_loader, desc="Caching"):
            images = images.to(device)
            all_labels.extend(labels_text)
            for name, m, arch in loaded_models:
                preds = model_forward_with_tta(m, arch, images, use_tta=True)
                all_model_preds[name].append(preds.cpu())

    # Concatenate [N_samples, T, C]
    cache = {name: torch.cat(all_model_preds[name], dim=0) for name, _, _ in loaded_models}
    torch.save({"cache": cache, "labels": all_labels}, "results/cached_val_preds.pt")
    print("✅ Cached predictions saved to results/cached_val_preds.pt")

    # Evaluate individual models
    print("\n--- SINGLE MODEL BASELINES ---")
    for name, _, _ in loaded_models:
        preds = cache[name]
        decoded = decode_with_confidence(preds, config.IDX2CHAR, use_format_prior=True)
        acc = sum(p == g for (p, _), g in zip(decoded, all_labels)) / len(all_labels) * 100
        print(f"  {name:12s}: {acc:.2f}% ({sum(p == g for (p, _), g in zip(decoded, all_labels))}/{len(all_labels)})")

    # Search combinations
    best_combo = None
    best_acc = 0.0
    best_decoded = None

    import itertools
    names = [name for name, _, _ in loaded_models]

    print("\n🔍 Grid searching ensemble combinations...")
    for k in range(2, len(names) + 1):
        for combo in itertools.combinations(names, k):
            # Equal weighting
            stacked = torch.stack([cache[c] for c in combo], dim=0)
            ens_preds = torch.logsumexp(stacked + math.log(1.0 / len(combo)), dim=0)
            decoded = decode_with_confidence(ens_preds, config.IDX2CHAR, use_format_prior=True)
            correct = sum(p == g for (p, _), g in zip(decoded, all_labels))
            acc = correct / len(all_labels) * 100
            if acc > best_acc:
                best_acc = acc
                best_combo = combo
                best_decoded = decoded
                print(f"  🎯 NEW PEAK: {acc:.2f}% ({correct}/{len(all_labels)}) | Models: {list(combo)}")

    print(f"\n🏆 BEST COMBINATION: {best_acc:.2f}% with {best_combo}")


if __name__ == "__main__":
    main()
