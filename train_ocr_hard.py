"""
Bước 2 (Stage 1b): Hard-Character Discriminative OCR Training on SR-Enhanced Domain.
Trains ResTranOCR with HardCharacterCTCLoss (Hard-Pair Focal Weighting + Prototype Angular
Margin Regularization) on top of the frozen 26.11 dB StackedSRNet (from Bước 1).
"""
import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.config import get_default_config
from src.data.dataset import MultiFrameDataset
from src.models.sr_restran import MultiFrameSRResTran
from src.training.ctc_loss import HardCharacterCTCLoss, HARD_CONFUSION_PAIRS
from src.training.trainer import decode_with_confidence
from src.utils.common import seed_everything
from scripts.eval_confusion import evaluate_confusion


@torch.no_grad()
def evaluate_ocr_hard(model, val_loader, idx2char, device, hard_criterion):
    """Evaluate Exact Match, CER, Margin Loss, and Hard-Pair Confusion count on 999 val tracks."""
    model.eval()
    preds_list = []
    gts_list = []

    for imgs, _, _, labels, _ in tqdm(val_loader, desc="Validating OCR", leave=False):
        imgs = imgs.to(device)
        with torch.amp.autocast(device_type=device.type) if device.type in ["cuda", "mps"] else torch.autocast("cpu", enabled=False):
            logits = model(imgs)
        decoded = decode_with_confidence(logits, idx2char)
        for (pred_str, _), gt_str in zip(decoded, labels):
            preds_list.append(pred_str)
            gts_list.append(gt_str)

    acc, cer, confusions = evaluate_confusion(preds_list, gts_list)
    margin_loss = hard_criterion.compute_margin_loss(model.ocr_net.head.weight).item()

    hard_pair_set = set(HARD_CONFUSION_PAIRS) | {(b, a) for a, b in HARD_CONFUSION_PAIRS}
    hard_errors = sum(cnt for pair, cnt in confusions.items() if pair in hard_pair_set)

    return acc * 100.0, cer * 100.0, margin_loss, hard_errors, confusions


def main():
    parser = argparse.ArgumentParser(description="Bước 2: Hard-Character Discriminative OCR Training")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--samples-per-epoch", type=int, default=4096, help="Tracks sampled per epoch")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate for OCR adaptation")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (0 on MPS, 2 on CUDA)")
    parser.add_argument("--max-hours", type=float, default=1.5, help="Hard time guard in hours (Rule 10)")
    parser.add_argument("--sr-weights", type=str, default="results/sr_net_best.pth", help="Warmed-up SR checkpoint")
    parser.add_argument("--ocr-init", type=str, default="results/restran_kaggle_gpu_best.pth", help="Initial OCR checkpoint")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    config = get_default_config()
    seed_everything(config.SEED)
    device = config.DEVICE
    num_workers = args.num_workers if args.num_workers is not None else (0 if device.type == "mps" else 2)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 BƯỚC 2 (STAGE 1B): HARD-CHARACTER DISCRIMINATIVE OCR TRAINING")
    print("=" * 70)
    print(f"   - Device             : {device}")
    print(f"   - Epochs             : {args.epochs} ({args.samples_per_epoch} hard-priority tracks/epoch)")
    print(f"   - Batch Size         : {args.batch_size}")
    print(f"   - Learning Rate      : {args.lr}")
    print(f"   - Warmed-up SR Ckpt  : {args.sr_weights} (FROZEN at 26.11 dB)")
    print(f"   - OCR Init Ckpt      : {args.ocr_init}")
    print(f"   - Max Hours Guard    : {args.max_hours}h (Rule 10)")
    print("=" * 70)

    train_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode="train",
        split_ratio=config.SPLIT_RATIO,
        img_height=16,
        img_width=64,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
        augmentation_level="light",
    )
    val_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode="val",
        split_ratio=config.SPLIT_RATIO,
        img_height=16,
        img_width=64,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
    )

    # Index hard-character real LR tracks (prioritizing Scenario-B + Hard Confusable Characters)
    hard_chars = {c for pair in HARD_CONFUSION_PAIRS for c in pair}
    sc_b_hard_indices = []
    sc_a_hard_indices = []
    other_real_indices = []

    for idx, sample in enumerate(train_ds.samples):
        if sample["is_synthetic"]:
            continue
        label = sample["label"]
        has_hard = any(c in hard_chars for c in label)
        is_sc_b = any("Scenario-B" in p for p in sample["paths"][:1])
        if is_sc_b and has_hard:
            sc_b_hard_indices.append(idx)
        elif has_hard:
            sc_a_hard_indices.append(idx)
        else:
            other_real_indices.append(idx)

    print(
        f"📊 Indexed Hard-Character Pool: "
        f"{len(sc_b_hard_indices)} Scenario-B Hard | "
        f"{len(sc_a_hard_indices)} Scenario-A Hard | "
        f"{len(other_real_indices)} Other Real"
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=num_workers,
    )

    # Build MultiFrameSRResTran and load Warmed-up SR + Pretrained OCR
    model = MultiFrameSRResTran(
        num_classes=config.NUM_CLASSES,
        transformer_heads=config.TRANSFORMER_HEADS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        transformer_ff_dim=config.TRANSFORMER_FF_DIM,
        dropout=config.TRANSFORMER_DROPOUT,
        use_stn=config.USE_STN,
        pretrained=True,
    ).to(device)

    if os.path.exists(args.sr_weights):
        print(f"📦 Loading Warmed-up SR weights from {args.sr_weights}")
        model.sr_net.load_state_dict(torch.load(args.sr_weights, map_location=device))
    else:
        raise FileNotFoundError(f"SR Warm-up checkpoint not found at {args.sr_weights}")

    if os.path.exists(args.ocr_init):
        print(f"📦 Loading initial OCR weights from {args.ocr_init}")
        ocr_state = torch.load(args.ocr_init, map_location=device)
        if any(k.startswith("ocr_net.") for k in ocr_state.keys()):
            ocr_state = {k.replace("ocr_net.", ""): v for k, v in ocr_state.items() if k.startswith("ocr_net.")}
        model.ocr_net.load_state_dict(ocr_state, strict=True)

    # Freeze SR completely during Bước 2 so 26.11 dB filters remain 100% intact
    for param in model.sr_net.parameters():
        param.requires_grad = False
    model.sr_net.eval()

    hard_criterion = HardCharacterCTCLoss(
        char2idx=config.CHAR2IDX,
        blank=0,
        alpha_hard=0.5,
        gamma_focal=0.5,
        margin=0.10,
        lambda_margin=2.0,
    )

    # Baseline evaluation before training
    init_acc, init_cer, init_margin, init_hard_err, _ = evaluate_ocr_hard(
        model, val_loader, config.IDX2CHAR, device, hard_criterion
    )
    print(
        f"🔍 [Initial Zero-Shot] Val Acc: {init_acc:.2f}% | CER: {init_cer:.2f}% | "
        f"Margin Loss: {init_margin:.5f} | Hard-Pair Errors: {init_hard_err}"
    )

    optimizer = optim.AdamW(model.ocr_net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = (args.samples_per_epoch + args.batch_size - 1) // args.batch_size
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        pct_start=0.2,
    )
    use_cuda_scaler = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_scaler)

    best_acc = init_acc
    best_cer = init_cer
    start_time = time.time()
    rng = random.Random(config.SEED)

    for epoch in range(args.epochs):
        elapsed_h = (time.time() - start_time) / 3600.0
        if args.max_hours is not None and elapsed_h >= args.max_hours:
            print(f"\n⏰ [TIME GUARD] Reached {elapsed_h:.2f}h >= {args.max_hours}h. Stopping safely.")
            break

        # Sample curriculum subset for this epoch (75% Scenario-B Hard, 25% Scenario-A Hard + Other)
        n_total = min(args.samples_per_epoch, len(sc_b_hard_indices) + len(sc_a_hard_indices))
        n_sc_b = min(int(n_total * 0.75), len(sc_b_hard_indices))
        n_sc_a = n_total - n_sc_b

        chosen_b = rng.sample(sc_b_hard_indices, n_sc_b)
        pool_a = sc_a_hard_indices + other_real_indices
        chosen_a = rng.sample(pool_a, min(n_sc_a, len(pool_a)))
        epoch_indices = chosen_b + chosen_a
        rng.shuffle(epoch_indices)

        train_loader = DataLoader(
            Subset(train_ds, epoch_indices),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=MultiFrameDataset.collate_fn,
            num_workers=num_workers,
        )

        model.train()
        model.sr_net.eval()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{args.epochs:02d}")

        for step, (imgs, targets, target_lengths, _, _) in enumerate(pbar):
            if step % 50 == 0 and args.max_hours is not None:
                if (time.time() - start_time) / 3600.0 >= args.max_hours:
                    print(f"\n⏰ [TIME GUARD] Mid-epoch limit ({args.max_hours}h) reached. Breaking safely.")
                    break

            imgs = imgs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type) if device.type in ["cuda", "mps"] else torch.autocast("cpu", enabled=False):
                preds = model(imgs)  # sr_net runs under torch.no_grad() automatically
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full((imgs.size(0),), preds.size(1), dtype=torch.long)
                loss = hard_criterion(
                    preds_permuted,
                    targets,
                    input_lengths,
                    target_lengths,
                    head_weight=model.ocr_net.head.weight,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_cuda_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.ocr_net.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.ocr_net.parameters(), max_norm=2.0)
                optimizer.step()

            if scheduler._step_count < scheduler.total_steps:
                scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_loss = train_loss / max(1, len(train_loader))
        val_acc, val_cer, val_margin, hard_err, confusions = evaluate_ocr_hard(
            model, val_loader, config.IDX2CHAR, device, hard_criterion
        )

        print(
            f"📈 Epoch {epoch + 1:02d}/{args.epochs:02d} | "
            f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | CER: {val_cer:.2f}% | "
            f"Margin: {val_margin:.5f} | Hard Errors: {hard_err}"
        )

        # Save latest
        torch.save(model.ocr_net.state_dict(), os.path.join(args.output_dir, "ocr_hard_pretrained_latest.pth"))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "sr_restran_latest.pth"))

        if val_acc > best_acc:
            best_acc = val_acc
            best_cer = val_cer
            ocr_best_path = os.path.join(args.output_dir, "ocr_hard_pretrained_best.pth")
            warmstart_path = os.path.join(args.output_dir, "flow2_warmstart_init.pth")
            flow2_best_path = os.path.join(args.output_dir, "sr_restran_best.pth")

            torch.save(model.ocr_net.state_dict(), ocr_best_path)
            torch.save(model.state_dict(), warmstart_path)
            torch.save(model.state_dict(), flow2_best_path)
            print(
                f"  ⭐ Saved New Best Hard-Character OCR -> {ocr_best_path} & {warmstart_path} "
                f"(Val Acc: {best_acc:.2f}% | CER: {best_cer:.2f}% | Hard Errors: {hard_err})"
            )

    total_mins = (time.time() - start_time) / 60.0
    print("=" * 70)
    print(f"✅ Bước 2 (Hard-Character OCR Training) Complete in {total_mins:.1f} mins!")
    print(f"🏆 Best Validation Exact Match: {best_acc:.2f}% | Best CER: {best_cer:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
