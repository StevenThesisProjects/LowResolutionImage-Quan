"""
Bước 3: Two-Stage Decoupled Training & Joint Fine-Tuning (train_decoupled.py)
Implements Two-Stage Protocol:
- Stage 1 (Frozen SR): OCR Fine-Tuning with HardCharacterCTCLoss on SR-enhanced domain.
- Stage 2 (Joint Fine-Tuning): End-to-End Joint Training (SR lr=1e-5, OCR lr=5e-5) with CTC + Auxiliary SRLoss.
Supports Rule 10 hard time guard (--max-hours), local smoke-testing, and MultiFrameSRResTran / MultiFrameSRViT.
"""
import os
import sys
import time
import glob
import json
import random
import argparse
from typing import Dict, List, Optional, Tuple

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import albumentations as A
import albumentations.pytorch as Ap

from configs.config import get_default_config
from src.models.sr_restran import MultiFrameSRResTran
from src.models.sr_vit import MultiFrameSRViT
from src.training.sr_loss import SRLoss
from src.training.ctc_loss import HardCharacterCTCLoss, HARD_CONFUSION_PAIRS
from src.training.trainer import decode_with_confidence
from src.data.transforms import get_degradation_transforms
from src.utils.common import seed_everything
from scripts.eval_confusion import evaluate_confusion


class JointDataset(Dataset):
    """
    Dataset for End-to-End Decoupled & Joint Training.
    Returns:
        - lr_tensor: [5, 3, lr_h, lr_w] in [-1, 1]
        - hr_tensor: [3, hr_h, hr_w] in [-1, 1] (center HR frame for synthetic samples, zero tensor for real LR)
        - targets: 1D tensor of char indices
        - target_len: scalar length
        - label: string plate text
        - track_id: string track identifier
        - is_synthetic: bool flag
    """
    def __init__(
        self,
        data_root: str,
        split_file: str,
        char2idx: Dict[str, int],
        mode: str = 'train',
        lr_h: int = 16,
        lr_w: int = 64,
        scale: int = 2,
        seed: int = 42,
    ):
        self.data_root = data_root
        self.mode = mode
        self.lr_h = lr_h
        self.lr_w = lr_w
        self.hr_h = lr_h * scale
        self.hr_w = lr_w * scale
        self.char2idx = char2idx
        self.seed = seed

        self.samples = []

        if not os.path.exists(split_file):
            os.makedirs(os.path.dirname(split_file), exist_ok=True)
            from src.data.dataset import MultiFrameDataset
            _ = MultiFrameDataset(
                root_dir=data_root,
                mode="train",
                val_split_file=split_file,
                split_ratio=0.9,
                seed=seed,
            )

        with open(split_file, 'r') as f:
            val_ids = set(json.load(f))

        from src.data.dataset import MultiFrameDataset

        abs_root = os.path.abspath(data_root)
        all_tracks = sorted(glob.glob(os.path.join(abs_root, "**", "track_*"), recursive=True))

        tracks = []
        for t in all_tracks:
            unique_id = MultiFrameDataset.get_track_unique_id(t)
            base_name = os.path.basename(t)
            is_val = unique_id in val_ids or (base_name in val_ids and "Scenario-B" in t)

            if mode == 'train' and not is_val:
                tracks.append(t)
            elif mode == 'val' and is_val:
                tracks.append(t)

        for t in tracks:
            json_path = os.path.join(t, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue

                track_id = os.path.basename(t)
                lr_files = sorted(glob.glob(os.path.join(t, "lr-*.png")) + glob.glob(os.path.join(t, "lr-*.jpg")))
                hr_files = sorted(glob.glob(os.path.join(t, "hr-*.png")) + glob.glob(os.path.join(t, "hr-*.jpg")))

                # Real LR samples
                if lr_files:
                    self.samples.append({
                        'paths': lr_files,
                        'hr_path': None,
                        'label': label,
                        'is_synthetic': False,
                        'track_id': track_id,
                        'scenario': 'Scenario-B' if 'Scenario-B' in t else 'Scenario-A',
                    })

                # Synthetic samples (only train mode)
                if hr_files and mode == 'train':
                    self.samples.append({
                        'paths': hr_files,
                        'hr_path': hr_files[len(hr_files) // 2],
                        'label': label,
                        'is_synthetic': True,
                        'track_id': track_id,
                        'scenario': 'Scenario-B' if 'Scenario-B' in t else 'Scenario-A',
                    })
            except Exception:
                pass

        # Transforms
        additional_targets = {f'image{i}': 'image' for i in range(1, 5)}
        self.degrade = get_degradation_transforms(4) if mode == 'train' else None

        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(self.lr_h, self.lr_w),
                A.ColorJitter(p=0.5),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                Ap.ToTensorV2()
            ], additional_targets=additional_targets, is_check_shapes=False)
        else:
            self.transform = A.Compose([
                A.Resize(self.lr_h, self.lr_w),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                Ap.ToTensorV2()
            ], additional_targets=additional_targets, is_check_shapes=False)

        self.hr_transform = A.Compose([
            A.Resize(self.hr_h, self.hr_w),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            Ap.ToTensorV2()
        ], is_check_shapes=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str, str, bool]:
        item = self.samples[idx]
        is_synthetic = item['is_synthetic']

        raw_images = []
        target_w = self.hr_w if is_synthetic else self.lr_w
        target_h = self.hr_h if is_synthetic else self.lr_h

        for p in item['paths']:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (target_w, target_h))
                raw_images.append(img)
        while len(raw_images) < 5:
            if len(raw_images) > 0:
                raw_images.append(raw_images[-1])
            else:
                raw_images.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
        raw_images = raw_images[:5]

        hr_img = np.zeros((self.hr_h, self.hr_w, 3), dtype=np.uint8)
        if is_synthetic and item['hr_path']:
            img = cv2.imread(item['hr_path'], cv2.IMREAD_COLOR)
            if img is not None:
                hr_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hr_img = cv2.resize(hr_img, (self.hr_w, self.hr_h))

        if is_synthetic and self.degrade:
            deg_targets = {f'image{i}': raw_images[i] for i in range(1, 5)}
            res = self.degrade(image=raw_images[0], **deg_targets)
            raw_images = [res['image']] + [res[f'image{i}'] for i in range(1, 5)]

        targets = {f'image{i}': raw_images[i] for i in range(1, 5)}
        res = self.transform(image=raw_images[0], **targets)
        lr_tensors = [res['image']] + [res[f'image{i}'] for i in range(1, 5)]
        lr_tensor = torch.stack(lr_tensors, dim=0)

        if is_synthetic:
            hr_tensor = self.hr_transform(image=hr_img)['image']
        else:
            hr_tensor = torch.zeros((3, self.hr_h, self.hr_w), dtype=torch.float32)

        label = item['label']
        target = [self.char2idx[c] for c in label if c in self.char2idx]
        if len(target) == 0:
            target = [0]

        return lr_tensor, hr_tensor, torch.tensor(target, dtype=torch.long), len(target), label, item['track_id'], is_synthetic

    @staticmethod
    def collate_fn(batch):
        lrs, hrs, targets, target_lengths, labels_text, track_ids, is_synth = zip(*batch)
        return (
            torch.stack(lrs, 0),
            torch.stack(hrs, 0),
            torch.cat(targets),
            torch.tensor(target_lengths, dtype=torch.long),
            labels_text,
            track_ids,
            torch.tensor(is_synth, dtype=torch.bool),
        )


@torch.no_grad()
def evaluate_model(model, val_loader, idx2char, device, val_limit=None):
    """Evaluates Exact Match, CER, Normalized Edit Accuracy, and hard-pair confusions."""
    model.eval()
    preds_list = []
    gts_list = []

    count = 0
    for lr_v, _, _, _, labels_v, _, _ in tqdm(val_loader, desc="Validating", leave=False):
        lr_v = lr_v.to(device)
        with torch.amp.autocast(device_type=device.type) if device.type in ["cuda", "mps"] else torch.autocast("cpu", enabled=False):
            logits_v = model(lr_v, return_sr=False)
        decoded = decode_with_confidence(logits_v, idx2char)
        for (pred_str, _), gt_str in zip(decoded, labels_v):
            preds_list.append(pred_str)
            gts_list.append(gt_str)
            count += 1
            if val_limit is not None and count >= val_limit:
                break
        if val_limit is not None and count >= val_limit:
            break

    acc, cer, confusions = evaluate_confusion(preds_list, gts_list)
    hard_pair_set = set(HARD_CONFUSION_PAIRS) | {(b, a) for a, b in HARD_CONFUSION_PAIRS}
    hard_errors = sum(cnt for pair, cnt in confusions.items() if pair in hard_pair_set)
    norm_edit_acc = max(0.0, 100.0 - cer * 100.0)

    return acc * 100.0, cer * 100.0, norm_edit_acc, hard_errors, len(preds_list)


def main():
    parser = argparse.ArgumentParser(description="Bước 3: Two-Stage Decoupled Training & Joint Fine-Tuning")
    parser.add_argument("--model", type=str, default="restran", choices=["restran", "sr_vit"], help="Model architecture")
    parser.add_argument("--frozen-epochs", type=int, default=5, help="Number of Stage 1 (Frozen SR) epochs")
    parser.add_argument("--joint-epochs", type=int, default=5, help="Number of Stage 2 (Joint Fine-Tuning) epochs")
    parser.add_argument("--samples-per-epoch", type=int, default=4096, help="Tracks sampled per epoch (0 for full dataset)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr-ocr-stage1", type=float, default=1.5e-4, help="Stage 1 OCR learning rate")
    parser.add_argument("--lr-ocr-stage2", type=float, default=5e-5, help="Stage 2 OCR learning rate")
    parser.add_argument("--lr-sr-stage2", type=float, default=1e-5, help="Stage 2 SR learning rate")
    parser.add_argument("--lambda-sr", type=float, default=0.1, help="Auxiliary SR loss weight in Stage 2")
    parser.add_argument("--lambda-margin", type=float, default=2.0, help="Hard-character angular margin weight")
    parser.add_argument("--max-hours", type=float, default=1.0, help="Rule 10 hard execution time cap in hours")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (0 on MPS, 2 on CUDA)")
    parser.add_argument("--init-weights", type=str, default="results/flow2_warmstart_init.pth", help="Path to warm-start checkpoint")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit validation samples (for fast dry run)")
    args = parser.parse_args()

    config = get_default_config()
    seed_everything(config.SEED)
    device = config.DEVICE
    num_workers = args.num_workers if args.num_workers is not None else (0 if device.type == "mps" else 2)
    os.makedirs(args.output_dir, exist_ok=True)

    total_epochs = args.frozen_epochs + args.joint_epochs

    print("=" * 75)
    print("🚀 BƯỚC 3: TWO-STAGE DECOUPLED TRAINING & JOINT FINE-TUNING")
    print("=" * 75)
    print(f"   - Architecture         : {args.model.upper()}")
    print(f"   - Device               : {device}")
    print(f"   - Stage 1 (Frozen SR)  : {args.frozen_epochs} epochs (LR_OCR={args.lr_ocr_stage1:.1e})")
    print(f"   - Stage 2 (Joint Train): {args.joint_epochs} epochs (LR_SR={args.lr_sr_stage2:.1e}, LR_OCR={args.lr_ocr_stage2:.1e}, λ_SR={args.lambda_sr})")
    print(f"   - Total Epochs         : {total_epochs}")
    print(f"   - Batch Size           : {args.batch_size}")
    print(f"   - Samples/Epoch        : {args.samples_per_epoch if args.samples_per_epoch > 0 else 'FULL DATASET'}")
    print(f"   - Rule 10 Hard Cap     : {args.max_hours:.2f} hours (Guaranteed hard stop)")
    print(f"   - Warm-start Checkpoint: {args.init_weights}")
    print("=" * 75)

    # 1. Datasets
    train_dataset = JointDataset(
        config.DATA_ROOT,
        config.VAL_SPLIT_FILE,
        config.CHAR2IDX,
        mode="train",
        lr_h=16,
        lr_w=64,
        scale=2,
    )
    val_dataset = JointDataset(
        config.DATA_ROOT,
        config.VAL_SPLIT_FILE,
        config.CHAR2IDX,
        mode="val",
        lr_h=16,
        lr_w=64,
        scale=2,
    )
    print(f"📦 Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # Partition train samples for balanced curriculum sampling
    hard_chars = {"8", "B", "0", "D", "5", "S", "2", "7", "6", "G", "V", "Y", "H", "M", "N", "W", "K", "T"}
    synth_indices = [i for i, s in enumerate(train_dataset.samples) if s['is_synthetic']]
    real_hard_indices = [i for i, s in enumerate(train_dataset.samples) if not s['is_synthetic'] and any(c in hard_chars for c in s['label'])]
    real_other_indices = [i for i, s in enumerate(train_dataset.samples) if not s['is_synthetic'] and not any(c in hard_chars for c in s['label'])]
    print(f"   Partitioning: Synthetic={len(synth_indices)}, Real Hard={len(real_hard_indices)}, Real Other={len(real_other_indices)}")

    val_batch_size = args.batch_size
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=JointDataset.collate_fn,
        num_workers=num_workers,
    )

    # 2. Model Instantiation
    if args.model == "sr_vit":
        model = MultiFrameSRViT(
            num_classes=config.NUM_CLASSES,
            sr_num_blocks=8,
            patch_h=config.SR_VIT_PATCH_H,
            patch_w=config.SR_VIT_PATCH_W,
            embed_dim=config.SR_VIT_EMBED_DIM,
            num_heads=config.SR_VIT_NUM_HEADS,
            num_layers=config.SR_VIT_NUM_LAYERS,
            ff_dim=config.SR_VIT_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            max_seq_len=(config.IMG_HEIGHT // config.SR_VIT_PATCH_H) * (config.IMG_WIDTH // config.SR_VIT_PATCH_W),
        ).to(device)
    else:
        model = MultiFrameSRResTran(
            num_classes=config.NUM_CLASSES,
            sr_num_frames=5,
            sr_in_channels=3,
            sr_base_channels=64,
            sr_num_blocks=8,
            sr_scale_factor=2,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            pretrained=True,
        ).to(device)

    # 3. Load Warm-start Weights
    if os.path.exists(args.init_weights):
        print(f"📦 Loading pre-warmed weights from {args.init_weights}...")
        ckpt = torch.load(args.init_weights, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"   Loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"⚠️ Initial weights {args.init_weights} not found. Checking individual checkpoints...")
        sr_path = os.path.join(args.output_dir, "sr_net_best.pth")
        if os.path.exists(sr_path):
            print(f"   Loading SR module from {sr_path}")
            model.sr_net.load_state_dict(torch.load(sr_path, map_location=device), strict=False)
        ocr_path = os.path.join(args.output_dir, "ocr_hard_pretrained_best.pth")
        if os.path.exists(ocr_path) and hasattr(model, 'ocr_net'):
            print(f"   Loading OCR module from {ocr_path}")
            model.ocr_net.load_state_dict(torch.load(ocr_path, map_location=device), strict=False)

    # 4. Losses
    ctc_loss_fn = HardCharacterCTCLoss(
        char2idx=config.CHAR2IDX,
        blank=0,
        alpha_hard=0.4,
        gamma_focal=0.5,
        margin=0.10,
        lambda_margin=args.lambda_margin,
    ).to(device)
    sr_loss_fn = SRLoss(w_mse=1.0, w_perceptual=0.1, w_edge=0.1, device=device).to(device)
    use_cuda_scaler = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_scaler)

    # Initial Validation
    print("\n🔍 Running initial validation...")
    init_acc, init_cer, init_edit, init_hard, n_eval = evaluate_model(
        model, val_loader, config.IDX2CHAR, device, val_limit=args.val_limit
    )
    print(f"   Baseline Val ({n_eval} tracks): Exact Match = {init_acc:.2f}% | CER = {init_cer:.2f}% | Edit Acc = {init_edit:.2f}% | Hard Errors = {init_hard}")

    best_val_acc = init_acc
    best_cer = init_cer
    save_path = os.path.join(args.output_dir, f"decoupled_{args.model}_best.pth")

    start_time = time.time()
    rng = random.Random(config.SEED)
    optimizer = None
    scheduler = None

    for epoch in range(total_epochs):
        elapsed_h = (time.time() - start_time) / 3600.0
        if args.max_hours is not None and elapsed_h >= args.max_hours:
            print(f"\n⏰ [TIME GUARD] Reached {elapsed_h:.2f}h >= {args.max_hours}h. Terminating training safely.")
            break

        is_stage1 = epoch < args.frozen_epochs

        # Configure Optimizer & Freeze / Unfreeze
        if epoch == 0 and args.frozen_epochs > 0:
            print("\n" + "=" * 70)
            print("🚀 STAGE 1: OCR FINE-TUNING (FROZEN SR)")
            print("=" * 70)
            for param in model.sr_net.parameters():
                param.requires_grad = False
            ocr_params = [p for n, p in model.named_parameters() if "sr_net" not in n]
            for param in ocr_params:
                param.requires_grad = True
            optimizer = optim.AdamW(ocr_params, lr=args.lr_ocr_stage1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.frozen_epochs), eta_min=1e-6)

        elif epoch == args.frozen_epochs:
            print("\n" + "=" * 70)
            print("🚀 STAGE 2: END-TO-END JOINT FINE-TUNING (UNFROZEN SR)")
            print("=" * 70)
            for param in model.sr_net.parameters():
                param.requires_grad = True
            ocr_params = [p for n, p in model.named_parameters() if "sr_net" not in n]
            optimizer = optim.AdamW([
                {"params": model.sr_net.parameters(), "lr": args.lr_sr_stage2, "weight_decay": 1e-4},
                {"params": ocr_params, "lr": args.lr_ocr_stage2, "weight_decay": 1e-4},
            ])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.joint_epochs), eta_min=1e-6)

        # Build DataLoader for this epoch
        if args.samples_per_epoch > 0 and args.samples_per_epoch < len(train_dataset):
            n_synth = min(args.samples_per_epoch // 2, len(synth_indices))
            n_real = args.samples_per_epoch - n_synth
            n_hard = min(int(n_real * 0.8), len(real_hard_indices))
            n_other = n_real - n_hard

            chosen_synth = rng.sample(synth_indices, n_synth)
            chosen_hard = rng.sample(real_hard_indices, n_hard)
            chosen_other = rng.sample(real_other_indices, min(n_other, len(real_other_indices)))
            epoch_indices = chosen_synth + chosen_hard + chosen_other
            rng.shuffle(epoch_indices)

            train_loader = DataLoader(
                Subset(train_dataset, epoch_indices),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=JointDataset.collate_fn,
                num_workers=num_workers,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=JointDataset.collate_fn,
                num_workers=num_workers,
            )

        model.train()
        if is_stage1:
            model.sr_net.eval()

        total_loss = 0.0
        total_ctc_loss = 0.0
        total_sr_loss = 0.0
        stage_name = "Stage 1 (Frozen SR)" if is_stage1 else "Stage 2 (Joint)"
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{total_epochs:02d} [{stage_name}]")

        for step, (lr, hr, targets, target_lengths, labels_text, track_ids, is_synth) in enumerate(pbar):
            if step % 50 == 0 and args.max_hours is not None:
                if (time.time() - start_time) / 3600.0 >= args.max_hours:
                    print(f"\n⏰ [TIME GUARD] Mid-epoch cap ({args.max_hours}h) reached. Breaking safely.")
                    break

            lr = lr.to(device)
            hr = hr.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            head_w = model.ocr_net.head.weight if hasattr(model, 'ocr_net') else model.head.weight

            with torch.amp.autocast(device_type=device.type) if device.type in ["cuda", "mps"] else torch.autocast("cpu", enabled=False):
                if is_stage1:
                    logits = model(lr, return_sr=False)
                    sr_image = None
                else:
                    logits, sr_image = model(lr, return_sr=True)

                preds_permuted = logits.permute(1, 0, 2)
                input_lengths = torch.full((lr.size(0),), logits.size(1), dtype=torch.long)

                loss_ctc = ctc_loss_fn(
                    preds_permuted,
                    targets,
                    input_lengths,
                    target_lengths,
                    head_weight=head_w,
                )

                if not is_stage1 and is_synth.any():
                    sr_image_synth = sr_image[is_synth]
                    hr_synth = hr[is_synth]
                    center_idx = sr_image_synth.size(1) // 2
                    loss_sr = sr_loss_fn(sr_image_synth[:, center_idx, :, :, :], hr_synth)
                else:
                    loss_sr = torch.tensor(0.0, device=device)

                loss = loss_ctc + (args.lambda_sr * loss_sr if not is_stage1 else 0.0)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_cuda_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            total_ctc_loss += loss_ctc.item()
            total_sr_loss += loss_sr.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "ctc": f"{loss_ctc.item():.3f}",
                "sr": f"{loss_sr.item():.3f}" if not is_stage1 else "0.0",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}",
            })

        avg_loss = total_loss / max(1, len(train_loader))
        scheduler.step()

        # Validation
        val_acc, val_cer, val_edit, val_hard, n_eval = evaluate_model(
            model, val_loader, config.IDX2CHAR, device, val_limit=args.val_limit
        )

        stage_tag = "STAGE 1" if is_stage1 else "STAGE 2"
        print(f"📊 [{stage_tag}] Epoch {epoch+1:02d}/{total_epochs:02d} -> Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | CER: {val_cer:.2f}% | Edit: {val_edit:.2f}% | Hard Err: {val_hard}")

        # Save latest checkpoint at every epoch
        latest_save_path = os.path.join(args.output_dir, f"decoupled_{args.model}_last.pth")
        torch.save(model.state_dict(), latest_save_path)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_cer = val_cer
            torch.save(model.state_dict(), save_path)
            print(f"  ⭐ Saved New Best Decoupled Model: {save_path} ({best_val_acc:.2f}%)")

    # Final summary metrics JSON
    total_elapsed_min = (time.time() - start_time) / 60.0
    summary = {
        "model": args.model,
        "total_epochs": total_epochs,
        "frozen_epochs": args.frozen_epochs,
        "joint_epochs": args.joint_epochs,
        "best_exact_match": best_val_acc,
        "best_cer": best_cer,
        "training_time_minutes": total_elapsed_min,
        "checkpoint": save_path,
    }
    metrics_path = os.path.join(args.output_dir, f"evaluation_metrics_decoupled_{args.model}.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "evaluation_metrics_decoupled.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print(f"🎉 DECOUPLED TRAINING FINISHED in {total_elapsed_min:.1f} mins!")
    print(f"   Best Val Exact Match : {best_val_acc:.2f}%")
    print(f"   Best Val CER         : {best_cer:.2f}%")
    print(f"   Checkpoint Saved     : {save_path}")
    print(f"   Metrics Saved        : {metrics_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
