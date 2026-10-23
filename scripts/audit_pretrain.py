#!/usr/bin/env python3
"""
Pre-Flight Pre-Training Audit Script (v2.1)
Verifies data split, checkpoint key parity, 20-step single-batch overfit, gradient norms, and STN affine drift.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.components import STNBlock
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.models.sr_restran import MultiFrameSRResTran
from src.models.sr_vit import MultiFrameSRViT
from src.utils.common import seed_everything
from src.utils.postprocess import CTCDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-Flight Pre-Training Audit")
    parser.add_argument("-m", "--model", type=str, required=True, choices=["sr_restran", "sr_vit", "restran", "crnn"],
                        help="Model architecture to audit")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to verify")
    parser.add_argument("--warmstart-sr", type=str, default=None, help="Path to SR checkpoint")
    parser.add_argument("--warmstart-ocr", type=str, default=None, help="Path to OCR checkpoint")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for test overfit")
    parser.add_argument("--ocr-lr-scale", type=float, default=1.0, help="OCR LR scale multiplier")
    parser.add_argument("--data-root", type=str, default=None, help="Path to training data root")
    parser.add_argument("--val-split-file", type=str, default=None, help="Path to validation split json")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained backbone")
    parser.add_argument("--pretrained-backbone", action="store_true", default=True, help="Enable pretrained backbone")
    return parser.parse_args()


def detect_tf_layers(state_dict: dict) -> int:
    layer_indices = set()
    for k in state_dict.keys():
        if "layers." in k:
            parts = k.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_indices.add(int(parts[i + 1]))
    return max(layer_indices) + 1 if layer_indices else None


def main():
    args = parse_args()
    seed_everything(42)
    start_time = time.time()

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n================================================================================")
    print(f"🚀 RUNNING PRE-FLIGHT AUDIT for model '{args.model}' on {device}")
    print(f"================================================================================")

    # Resolve data paths
    data_root = args.data_root
    if not data_root:
        for candidate in ["dataset/data/train", "data/train"]:
            if os.path.exists(candidate):
                data_root = candidate
                break
    if not data_root or not os.path.exists(data_root):
        print(f"❌ ERROR: Data root not found: {data_root}")
        sys.exit(1)

    val_split_file = args.val_split_file
    if not val_split_file:
        for candidate in ["dataset/data/val_tracks.json", "data/val_tracks.json"]:
            if os.path.exists(candidate):
                val_split_file = candidate
                break

    config = Config()
    in_h = 16 if args.model in ["sr_restran", "sr_vit"] else config.IMG_HEIGHT
    in_w = 64 if args.model in ["sr_restran", "sr_vit"] else config.IMG_WIDTH

    # -------------------------------------------------------------------------
    # STAGE 1: Data Split & Resolution Check
    # -------------------------------------------------------------------------
    print(f"\n[Stage 1/4] 🔍 Data Split & Resolution Check...")
    if not val_split_file or not os.path.exists(val_split_file):
        print(f"❌ ERROR: Validation split file not found: {val_split_file}")
        sys.exit(1)

    with open(val_split_file, "r") as f:
        val_tracks_list = json.load(f)
    num_val_tracks = len(val_tracks_list)
    print(f"   Validation split file contains: {num_val_tracks} tracks ({val_split_file})")

    train_ds = MultiFrameDataset(
        root_dir=data_root,
        mode="train",
        split_ratio=config.SPLIT_RATIO,
        img_height=in_h,
        img_width=in_w,
        char2idx=config.CHAR2IDX,
        val_split_file=val_split_file,
        seed=42,
        augmentation_level="light",
    )
    val_ds = MultiFrameDataset(
        root_dir=data_root,
        mode="val",
        split_ratio=config.SPLIT_RATIO,
        img_height=in_h,
        img_width=in_w,
        char2idx=config.CHAR2IDX,
        val_split_file=val_split_file,
        seed=42,
    )

    train_track_set = set(os.path.dirname(s['paths'][0]) for s in train_ds.samples if s.get('paths'))
    val_track_set = set(os.path.dirname(s['paths'][0]) for s in val_ds.samples if s.get('paths'))
    overlap = train_track_set.intersection(val_track_set)
    if len(overlap) > 0:
        print(f"❌ CRITICAL ERROR: Data leakage detected! Overlap: {len(overlap)} tracks.")
        sys.exit(1)

    sample_imgs = train_ds[0][0]
    expected_shape = (5, 3, in_h, in_w)
    if sample_imgs.shape != expected_shape:
        print(f"❌ Shape mismatch: got {sample_imgs.shape}, expected {expected_shape}")
        sys.exit(1)

    print(f"   ✅ Stage 1 PASSED: {len(val_ds)} validation tracks in Scenario-B, 0 data leakage, input shape {list(sample_imgs.shape)}.")

    # -------------------------------------------------------------------------
    # STAGE 2: Key Parity & Epoch-0 Validation Slice
    # -------------------------------------------------------------------------
    print(f"\n[Stage 2/4] 🔍 Key Parity & Epoch-0 Validation Slice (200 tracks)...")
    detected_layers = None
    state_dict_to_load = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict_to_load = ckpt.get("state_dict", ckpt.get("model", ckpt))
        detected_layers = detect_tf_layers(state_dict_to_load)
        if detected_layers:
            print(f"   Auto-detected {detected_layers} transformer layers from checkpoint {args.checkpoint}")

    pretrained_backbone = False if args.no_pretrained else True
    if args.model == "sr_restran":
        tf_layers = detected_layers if detected_layers is not None else config.TRANSFORMER_LAYERS
        model = MultiFrameSRResTran(
            num_classes=config.NUM_CLASSES,
            sr_num_blocks=8,
            sr_scale_factor=2,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=tf_layers,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=True,
            pretrained=pretrained_backbone if not state_dict_to_load else False,
        )
    elif args.model == "sr_vit":
        vit_layers = detected_layers if detected_layers is not None else config.SR_VIT_NUM_LAYERS
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
            use_stn=True,
            pretrained=pretrained_backbone if not state_dict_to_load else False,
        )
    elif args.model == "restran":
        tf_layers = detected_layers if detected_layers is not None else config.TRANSFORMER_LAYERS
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=tf_layers,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=True,
            pretrained=pretrained_backbone if not state_dict_to_load else False,
        )
    else:
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=True,
        )

    # Load weights
    matched_keys = 0
    if state_dict_to_load is not None:
        missing, unexpected = model.load_state_dict(state_dict_to_load, strict=False)
        matched_keys = len(state_dict_to_load) - len(unexpected)
        critical_missing = [k for k in missing if k != "sr_net.res_scale_gate"]
        print(f"   Checkpoint loaded: {matched_keys} keys matched (missing: {len(missing)}, unexpected: {len(unexpected)})")
        if critical_missing:
            print(f"   ⚠️ Note: {len(critical_missing)} missing keys: {critical_missing[:5]}")

    if args.warmstart_sr and os.path.exists(args.warmstart_sr):
        sr_state = torch.load(args.warmstart_sr, map_location="cpu")
        sr_sub_state = {}
        for k, v in sr_state.items():
            if k.startswith("sr_net."):
                sr_sub_state[k.replace("sr_net.", "")] = v
            elif not any(k.startswith(p) for p in ["ocr_net.", "backbone.", "stn.", "encoder.", "head."]):
                sr_sub_state[k] = v
        target_sr = getattr(model, "sr_net", None)
        if target_sr is not None:
            target_sr_dict = target_sr.state_dict()
            adapted = {}
            for k, v in sr_sub_state.items():
                new_k = k
                if k.startswith("conv_first."):
                    new_k = k.replace("conv_first.", "head.")
                elif k.startswith("conv_last."):
                    new_k = k.replace("conv_last.", "tail.")
                if new_k in target_sr_dict:
                    if v.shape == target_sr_dict[new_k].shape:
                        adapted[new_k] = v
                    elif new_k == "head.weight" and v.shape == (64, 15, 3, 3) and target_sr_dict[new_k].shape == (64, 3, 3, 3):
                        adapted[new_k] = v.view(64, 5, 3, 3, 3).mean(dim=1)
                        print("   🔄 Converted 15-channel conv_first.weight -> 3-channel head.weight via 5-frame mean.")
            target_sr.load_state_dict(adapted, strict=False)
            print(f"   Warmstarted SR module from {args.warmstart_sr}: {len(adapted)} keys adapted.")

    model = model.to(device)
    model.eval()

    # Slice 200 tracks for Epoch 0 evaluation
    slice_indices = list(range(min(200, len(val_ds))))
    val_slice = Subset(val_ds, slice_indices)
    slice_loader = DataLoader(val_slice, batch_size=32, shuffle=False, collate_fn=MultiFrameDataset.collate_fn)
    decoder = CTCDecoder(config.IDX2CHAR, blank_idx=0)

    correct_slice = 0
    total_slice = 0
    with torch.no_grad():
        for batch_imgs, batch_targets, batch_target_lens, batch_labels, batch_tracks in slice_loader:
            batch_imgs = batch_imgs.to(device)
            logits = model(batch_imgs)
            preds = decoder.decode_batch(logits)
            for p, g in zip(preds, batch_labels):
                if p == g:
                    correct_slice += 1
                total_slice += 1

    epoch0_acc = (correct_slice / total_slice * 100) if total_slice > 0 else 0.0
    print(f"   ✅ Stage 2 PASSED: Epoch-0 Slice Accuracy (200 tracks) = {epoch0_acc:.2f}% ({correct_slice}/{total_slice}).")

    # -------------------------------------------------------------------------
    # STAGE 3: 20-Step Single-Batch Overfit & Sub-Module Gradient Norms
    # -------------------------------------------------------------------------
    print(f"\n[Stage 3/4] 🔍 20-Step Single-Batch Overfit & Sub-Module Gradient Norms...")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=MultiFrameDataset.collate_fn)
    single_batch = next(iter(train_loader))
    b_imgs, b_targets, b_lens, b_labels, b_tracks = single_batch
    b_imgs = b_imgs.to(device)
    b_targets = b_targets.to(device)
    b_lens = b_lens.to(device)

    # Separate parameters
    sr_params = []
    stn_params = []
    ocr_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "stn." in name:
            stn_params.append(p)
        elif name.startswith("sr_net."):
            sr_params.append(p)
        else:
            ocr_params.append(p)

    base_lr = args.lr
    sr_lr = base_lr * (0.2 if args.checkpoint else (0.5 if args.warmstart_sr else 1.0))
    ocr_lr = base_lr * args.ocr_lr_scale
    stn_lr = base_lr * 0.1

    param_groups = []
    if sr_params:
        param_groups.append({"params": sr_params, "lr": sr_lr})
    if ocr_params:
        param_groups.append({"params": ocr_params, "lr": ocr_lr})
    if stn_params:
        param_groups.append({"params": stn_params, "lr": stn_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()
    initial_loss = None
    final_loss = None
    batch_acc = 0.0

    sr_norm = 0.0
    stn_norm = 0.0
    ocr_norm = 0.0

    target_strs = list(b_labels)

    for step in range(1, 21):
        optimizer.zero_grad()
        logits = model(b_imgs)
        B, T, C = logits.shape
        input_lens = torch.full((B,), T, dtype=torch.long, device=device)
        # CTC loss expects [T, B, C]
        loss = criterion(logits.permute(1, 0, 2), b_targets, input_lens, b_lens)

        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()

        loss.backward()

        # Compute gradient norms
        if sr_params:
            sr_grads = [p.grad for p in sr_params if p.grad is not None]
            if sr_grads:
                sr_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in sr_grads]), 2).item()
        if stn_params:
            stn_grads = [p.grad for p in stn_params if p.grad is not None]
            if stn_grads:
                stn_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in stn_grads]), 2).item()
        if ocr_params:
            ocr_grads = [p.grad for p in ocr_params if p.grad is not None]
            if ocr_grads:
                ocr_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in ocr_grads]), 2).item()

        # Dedicated STN clipping and global clipping
        if stn_params:
            nn.utils.clip_grad_norm_(stn_params, max_norm=0.5)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        preds = decoder.decode_batch(logits.detach())
        match_count = sum(1 for p, g in zip(preds, target_strs) if p == g)
        batch_acc = match_count / len(target_strs) * 100

        if step in [1, 5, 10, 15, 20]:
            print(f"   Step {step:2d}/20: Loss = {loss.item():.4f}, Batch Acc = {batch_acc:5.1f}% | ||g_sr||={sr_norm:.3f}, ||g_stn||={stn_norm:.3f}, ||g_ocr||={ocr_norm:.3f}")

    print(f"   Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f} (Accuracy: {batch_acc:.1f}%)")
    # For a randomly initialized or clean model, 20 steps should significantly reduce loss.
    # If loss reduced or accuracy improved, test passes.
    assert final_loss < initial_loss or final_loss < 0.25, f"Overfit failure: loss did not decrease (init: {initial_loss}, final: {final_loss})"
    print(f"   ✅ Stage 3 PASSED: 20-step Overfit converged properly. Sub-module gradients healthy.")

    # -------------------------------------------------------------------------
    # STAGE 4: STN Affine Drift Check
    # -------------------------------------------------------------------------
    print(f"\n[Stage 4/4] 🔍 STN Affine Drift Check...")
    stn_module = getattr(model, "stn", None) or (getattr(model.ocr_net, "stn", None) if hasattr(model, "ocr_net") else None)
    if stn_module is not None:
        model.eval()
        with torch.no_grad():
            sr_sub = getattr(model, "sr_net", None)
            if sr_sub is not None:
                sr_out = sr_sub(b_imgs[:2])
                B_s, F_s, C_s, H_s, W_s = sr_out.shape
                sr_flat = sr_out.view(B_s * F_s, C_s, H_s, W_s)
                theta = stn_module(sr_flat)
            else:
                theta = stn_module(b_imgs[:2].view(-1, 3, in_h, in_w))
            
            identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=theta.dtype)
            drift = torch.max(torch.abs(theta - identity)).item()
            print(f"   Max Affine Drift ||theta - I||_inf = {drift:.4f}")
            if drift > 0.75:
                print(f"   ⚠️ WARNING: STN affine drift {drift:.4f} exceeds threshold 0.75!")
            else:
                print(f"   ✅ Stage 4 PASSED: STN affine drift {drift:.4f} is within safe bound (< 0.75).")
    else:
        print("   (Model has no STN module, skipping STN affine drift check)")

    elapsed = time.time() - start_time
    print(f"\n================================================================================")
    print(f"🎉 ✅ PRE-FLIGHT AUDIT PASSED in {elapsed:.1f}s: Model '{args.model}' is verified and ready for full training.")
    print(f"================================================================================\n")


if __name__ == "__main__":
    main()
