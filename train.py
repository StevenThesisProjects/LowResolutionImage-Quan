#!/usr/bin/env python3
"""Main entry point for OCR training pipeline."""
import argparse
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys

import torch
from torch.utils.data import DataLoader

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.training.trainer import Trainer
from src.utils.common import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Frame OCR for License Plate Recognition"
    )
    parser.add_argument(
        "-n", "--experiment-name", type=str, default=None,
        help="Experiment name for checkpoint/submission files (default: from config)"
    )
    parser.add_argument(
        "-m", "--model", type=str, choices=["crnn", "restran", "sr_restran", "sr_vit"], default=None,
        help="Model architecture: 'crnn', 'restran', 'sr_restran' or 'sr_vit' (default: from config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for training (default: from config)"
    )
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=None,
        dest="learning_rate",
        help="Learning rate (default: from config)"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for training data (default: from config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of data loader workers (default: from config)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=None,
        help="LSTM hidden size for CRNN (default: from config)"
    )
    parser.add_argument(
        "--transformer-heads", type=int, default=None,
        help="Number of transformer attention heads (default: from config)"
    )
    parser.add_argument(
        "--transformer-layers", type=int, default=None,
        help="Number of transformer encoder layers (default: from config)"
    )
    parser.add_argument(
        "--aug-level",
        type=str,
        choices=["full", "light"],
        default=None,
        help="Augmentation level for training data (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save checkpoints and submission files (default: results/)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Max gradient norm for clipping (default: from config)",
    )
    parser.add_argument(
        "--no-stn",
        action="store_true",
        help="Disable Spatial Transformer Network (STN) alignment",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights for ResNet-34 backbone",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        default=True,
        help="Enable ImageNet pretrained weights for ResNet-34 backbone (default: True)",
    )
    parser.add_argument(
        "--submission-mode",
        action="store_true",
        help="Train on full dataset and generate submission file for test data",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Abort immediately if no GPU (CUDA/MPS) is detected",
    )
    parser.add_argument(
        "--freeze-sr",
        action="store_true",
        help="Freeze Super-Resolution module (requires_grad=False) to prevent gradient distortion (Bug 4 fix)",
    )
    parser.add_argument(
        "--warmstart-ocr",
        type=str,
        default=None,
        help="Path to pretrained OCR checkpoint (.pth) to initialize OCR backbone/heads",
    )
    parser.add_argument(
        "--warmstart-sr",
        type=str,
        default=None,
        help="Path to pretrained Super-Resolution checkpoint (.pth) to initialize SR module",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to full model checkpoint (.pth) to resume training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: run 5 training batches and sample validation to verify pipeline and measure batch speed (Rule 8)",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Hard execution time limit in hours. Automatically breaks and saves checkpoints if exceeded.",
    )
    parser.add_argument(
        "--use-hard-loss",
        action="store_true",
        help="Use HardCharacterCTCLoss instead of standard nn.CTCLoss",
    )
    parser.add_argument(
        "--ocr-lr-scale",
        type=float,
        default=1.0,
        help="Scale multiplier for OCR parameter learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--val-split-file",
        type=str,
        default=None,
        help="Path to validation tracks JSON file",
    )
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Initialize config with CLI overrides
    config = Config()
    
    # Map CLI arguments to config attributes
    arg_to_config = {
        'experiment_name': 'EXPERIMENT_NAME',
        'model': 'MODEL_TYPE',
        'epochs': 'EPOCHS',
        'batch_size': 'BATCH_SIZE',
        'learning_rate': 'LEARNING_RATE',
        'data_root': 'DATA_ROOT',
        'val_split_file': 'VAL_SPLIT_FILE',
        'seed': 'SEED',
        'num_workers': 'NUM_WORKERS',
        'hidden_size': 'HIDDEN_SIZE',
        'transformer_heads': 'TRANSFORMER_HEADS',
        'transformer_layers': 'TRANSFORMER_LAYERS',
        'grad_clip': 'GRAD_CLIP',
        'max_hours': 'MAX_HOURS',
    }
    
    for arg_name, config_name in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)
    
    # Default EXPERIMENT_NAME to model if not explicitly specified
    if args.experiment_name is None and args.model is not None:
        config.EXPERIMENT_NAME = args.model
    
    # Special cases
    if args.aug_level is not None:
        config.AUGMENTATION_LEVEL = args.aug_level
    
    if args.no_stn:
        config.USE_STN = False
    
    if args.no_pretrained:
        config.BACKBONE_PRETRAINED = False
        
    if args.freeze_sr:
        config.FREEZE_SR = True
        
    config.USE_HARD_LOSS = bool(args.use_hard_loss)
    config.WARMSTART_OCR = bool(args.warmstart_ocr)
    config.WARMSTART_SR = bool(args.warmstart_sr)
    config.HAS_CHECKPOINT = bool(args.checkpoint)
    config.OCR_LR_SCALE = getattr(args, 'ocr_lr_scale', 1.0)

    if args.dry_run:
        config.DRY_RUN = True
        config.EPOCHS = 1
        print("\n🧪 DRY-RUN MODE ACTIVATED (Rule 8): Running 5 train batches and sample val batches to estimate runtime.")
    
    # Output directory
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    seed_everything(config.SEED)
    
    print(f"🚀 Configuration:")
    print(f"   EXPERIMENT: {config.EXPERIMENT_NAME}")
    print(f"   MODEL: {config.MODEL_TYPE}")
    print(f"   USE_STN: {config.USE_STN}")
    print(f"   DATA_ROOT: {config.DATA_ROOT}")
    print(f"   EPOCHS: {config.EPOCHS}")
    print(f"   BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   DEVICE: {config.DEVICE}")
    print(f"   SUBMISSION_MODE: {args.submission_mode}")
    
    if args.require_gpu:
        if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            print("\n❌ CRITICAL ERROR: --require-gpu flag is set, but PyTorch could not detect any GPU (CUDA/MPS)!")
            print("   Execution aborted immediately to prevent running on CPU.\n")
            sys.exit(1)

    # Validate data path
    if not os.path.exists(config.DATA_ROOT):
        print(f"❌ ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)

    # Common dataset parameters
    in_h = 16 if config.MODEL_TYPE in ["sr_restran", "sr_vit"] else config.IMG_HEIGHT
    in_w = 64 if config.MODEL_TYPE in ["sr_restran", "sr_vit"] else config.IMG_WIDTH
    common_ds_params = {
        'split_ratio': config.SPLIT_RATIO,
        'img_height': in_h,
        'img_width': in_w,
        'char2idx': config.CHAR2IDX,
        'val_split_file': config.VAL_SPLIT_FILE,
        'seed': config.SEED,
        'augmentation_level': config.AUGMENTATION_LEVEL,
    }
    
    # Create datasets based on mode
    if args.submission_mode:
        print("\n📌 SUBMISSION MODE ENABLED")
        print("   - Training on FULL dataset (no validation split)")
        print("   - Will generate predictions for test data after training\n")
        
        # Create training dataset with full_train=True
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            full_train=True,
            **common_ds_params
        )
        
        # Create test dataset if test data exists
        test_loader = None
        if os.path.exists(config.TEST_DATA_ROOT):
            test_ds = MultiFrameDataset(
                root_dir=config.TEST_DATA_ROOT,
                mode='val',
                img_height=in_h,
                img_width=in_w,
                char2idx=config.CHAR2IDX,
                seed=config.SEED,
                is_test=True,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )
        else:
            print(f"⚠️ WARNING: Test data not found at {config.TEST_DATA_ROOT}")
        
        val_loader = None
    else:
        # Normal training/validation split mode
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            **common_ds_params
        )
        
        val_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='val',
            **common_ds_params
        )
        
        val_loader = None
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=(config.DEVICE.type == 'cuda')
            )
        else:
            print("⚠️ WARNING: Validation dataset is empty.")
        
        test_loader = None
    
    if len(train_ds) == 0:
        print("❌ Training dataset is empty!")
        sys.exit(1)

    # Create training data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE.type == 'cuda')
    )

    # Auto-detect transformer layers from checkpoint if provided
    detected_tf_layers = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            ckpt_peek = torch.load(args.checkpoint, map_location="cpu")
            if isinstance(ckpt_peek, dict):
                sd_peek = ckpt_peek.get("state_dict", ckpt_peek.get("model", ckpt_peek))
                layer_indices = set()
                for k in sd_peek.keys():
                    if "layers." in k:
                        parts = k.split(".")
                        for i, part in enumerate(parts):
                            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                                layer_indices.add(int(parts[i + 1]))
                if layer_indices:
                    detected_tf_layers = max(layer_indices) + 1
                    print(f"🔍 Auto-detected {detected_tf_layers} transformer layers from resume checkpoint {args.checkpoint}")
        except Exception as e:
            pass

    # Initialize model based on config
    if config.MODEL_TYPE == "sr_vit":
        from src.models.sr_vit import MultiFrameSRViT
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
            pretrained=getattr(config, 'BACKBONE_PRETRAINED', True),
        ).to(config.DEVICE)
    elif config.MODEL_TYPE == "sr_restran":
        from src.models.sr_restran import MultiFrameSRResTran
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
            pretrained=getattr(config, 'BACKBONE_PRETRAINED', True),
        ).to(config.DEVICE)
    elif config.MODEL_TYPE == "restran":
        tf_layers = detected_tf_layers if detected_tf_layers is not None else config.TRANSFORMER_LAYERS
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=tf_layers,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            pretrained=getattr(config, 'BACKBONE_PRETRAINED', True),
        ).to(config.DEVICE)
    else:
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
        ).to(config.DEVICE)
    
    # 1. Full checkpoint loading
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"📦 Loading full model checkpoint from {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location=config.DEVICE)
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            matched = len(ckpt) - len(unexpected)
            print(f"   ✅ Checkpoint loaded: {matched} keys matched (missing: {len(missing)}, unexpected: {len(unexpected)})")
        else:
            print(f"❌ ERROR: Checkpoint not found at {args.checkpoint}")
            sys.exit(1)

    # 2. Warmstart SR module
    sr_loaded_ok = False
    if args.warmstart_sr:
        if os.path.exists(args.warmstart_sr):
            print(f"📦 Warm-starting SR module from {args.warmstart_sr}")
            sr_state = torch.load(args.warmstart_sr, map_location=config.DEVICE)
            # If state has 'sr_net.' prefix, extract only sr_net keys
            sr_sub_state = {}
            for k, v in sr_state.items():
                if k.startswith("sr_net."):
                    sr_sub_state[k.replace("sr_net.", "")] = v
                elif not any(k.startswith(p) for p in ["ocr_net.", "backbone.", "stn.", "encoder.", "head."]):
                    sr_sub_state[k] = v
            
            target_sr = getattr(model, 'sr_net', None)
            if target_sr is not None:
                target_sr_dict = target_sr.state_dict()
                adapted_sr_state = {}
                skipped_shapes = 0
                for k, v in sr_sub_state.items():
                    new_k = k
                    if k.startswith("conv_first."):
                        new_k = k.replace("conv_first.", "head.")
                    elif k.startswith("conv_last."):
                        new_k = k.replace("conv_last.", "tail.")
                    if new_k in target_sr_dict:
                        if v.shape == target_sr_dict[new_k].shape:
                            adapted_sr_state[new_k] = v
                        elif (
                            new_k == "head.weight"
                            and v.shape == (64, 15, 3, 3)
                            and target_sr_dict[new_k].shape == (64, 3, 3, 3)
                        ):
                            # Quy đổi bộ lọc 15 kênh (5 frames x 3 RGB) sang 3 kênh bằng trung bình 5 frames
                            adapted_sr_state[new_k] = v.view(64, 5, 3, 3, 3).mean(dim=1)
                            print("   🔄 Converted 15-channel conv_first.weight [64,15,3,3] -> 3-channel head.weight [64,3,3,3] via 5-frame mean.")
                        else:
                            skipped_shapes += 1
                if adapted_sr_state:
                    missing, unexpected = target_sr.load_state_dict(adapted_sr_state, strict=False)
                    sr_loaded_ok = len(adapted_sr_state) >= 10
                    print(f"   ✅ SR module weights loaded: {len(adapted_sr_state)} keys matched (skipped {skipped_shapes} shape-mismatched keys)")
                else:
                    print(f"   ⚠️ WARNING: No compatible 3-channel SR weights found in {args.warmstart_sr} (skipped {skipped_shapes} incompatible keys).")
            else:
                print(f"   ⚠️ Model {config.MODEL_TYPE} does not have a 'sr_net' submodule. Skipping SR warmstart.")
        else:
            print(f"❌ ERROR: SR checkpoint not found at {args.warmstart_sr}")
            sys.exit(1)

    # 3. Warmstart OCR module
    if args.warmstart_ocr:
        if os.path.exists(args.warmstart_ocr):
            print(f"📦 Warm-starting OCR module from {args.warmstart_ocr}")
            ocr_state = torch.load(args.warmstart_ocr, map_location=config.DEVICE)
            
            target_ocr = getattr(model, 'ocr_net', None)
            if target_ocr is not None:
                # Target is model.ocr_net (e.g. MultiFrameSRResTran)
                ocr_sub_state = {}
                for k, v in ocr_state.items():
                    clean_k = k.replace("ocr_net.", "")
                    ocr_sub_state[clean_k] = v
                missing, unexpected = target_ocr.load_state_dict(ocr_sub_state, strict=False)
                matched = len(ocr_sub_state) - len(unexpected)
                print(f"   ✅ OCR module weights loaded: {matched} keys matched into ocr_net (missing: {len(missing)})")
            elif hasattr(model, 'conv_stem'):
                # Target is MultiFrameSRViT: map ResNet34 conv1..layer3 into conv_stem.0..6
                stem_map = {
                    "backbone.conv1.": "conv_stem.0.",
                    "backbone.bn1.": "conv_stem.1.",
                    "backbone.layer1.": "conv_stem.4.",
                    "backbone.layer2.": "conv_stem.5.",
                    "backbone.layer3.": "conv_stem.6.",
                    "stn.": "stn.",
                }
                mapped_state = {}
                for k, v in ocr_state.items():
                    clean_k = k[len("ocr_net."):] if k.startswith("ocr_net.") else k
                    for src_prefix, dst_prefix in stem_map.items():
                        if clean_k.startswith(src_prefix):
                            mapped_state[dst_prefix + clean_k[len(src_prefix):]] = v
                            break
                missing, unexpected = model.load_state_dict(mapped_state, strict=False)
                matched = len(mapped_state) - len(unexpected)
                print(f"   ✅ Mapped {matched} ResNet-34 stem/STN keys into MultiFrameSRViT (missing: {len(missing)})")
            else:
                # Target is root model (e.g. ResTranOCR)
                missing, unexpected = model.load_state_dict(ocr_state, strict=False)
                matched = len(ocr_state) - len(unexpected)
                print(f"   ✅ Model weights loaded directly: {matched} keys matched (missing: {len(missing)})")
        else:
            print(f"❌ ERROR: OCR checkpoint not found at {args.warmstart_ocr}")
            sys.exit(1)

    # 4. Freeze SR module if requested (Bug 4 fix)
    if args.freeze_sr:
        if hasattr(model, 'sr_net'):
            for name, p in model.sr_net.named_parameters():
                if name == "res_scale_gate":
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            model.sr_net.eval()
            config.FREEZE_SR = True
            print("❄️ Super-Resolution Module FROZEN (requires_grad = False, res_scale_gate trainable). Protecting SR filters from gradient distortion.")
        else:
            print(f"⚠️ Model {config.MODEL_TYPE} does not have 'sr_net'. --freeze-sr ignored.")

    # Res scale gate protection for untrained SR with warmstart OCR
    if args.warmstart_ocr and hasattr(model, 'sr_net'):
        if not sr_loaded_ok and not args.checkpoint:
            if hasattr(model.sr_net, 'res_scale_gate'):
                model.sr_net.res_scale_gate.data.fill_(0.05)
                print("   🛡️ Random-SR Protection: initialized res_scale_gate to 0.05 to preserve warmstart OCR accuracy at Epoch 0.")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model ({config.MODEL_TYPE}): {total_params:,} total params, {trainable_params:,} trainable")

    # Initialize trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR
    )
    
    trainer.fit()
    
    # Run test inference in submission mode
    if args.submission_mode and test_loader is not None:
        print("\n" + "="*60)
        print("📝 GENERATING SUBMISSION FILE")
        print("="*60)
        
        # Load best checkpoint if it exists
        exp_name = config.EXPERIMENT_NAME
        best_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_best.pth")
        if os.path.exists(best_model_path):
            print(f"📦 Loading best checkpoint: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        else:
            print("⚠️ No best checkpoint found, using final model weights")
        
        # Run inference on test data
        trainer.predict_test(test_loader, output_filename=f"submission_{exp_name}_final.txt")


if __name__ == "__main__":
    main()
