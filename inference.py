"""Inference script for generating license plate predictions on public test dataset."""
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


def parse_args():
    parser = argparse.ArgumentParser(description="Run LPR Inference on Test Data")
    parser.add_argument(
        "--model",
        type=str,
        default="end2end",
        choices=["crnn", "restran", "end2end"],
        help="Model architecture to evaluate",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/public_test",
        help="Path to public test dataset directory",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/submission_crnn_baseline_full.txt",
        help="Path to output submission file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    seed_everything(config.SEED)
    
    device = config.DEVICE
    print(f"🚀 Running Inference using device: {device}")
    
    # 1. Verify test data directory
    test_dir = args.test_dir
    if not os.path.exists(test_dir):
        print(f"❌ ERROR: Test directory not found at {test_dir}")
        sys.exit(1)
        
    # 2. Load test dataset
    print(f"📁 Loading test data from: {test_dir}")
    in_h = 16 if args.model == "end2end" else config.IMG_HEIGHT
    in_w = 64 if args.model == "end2end" else config.IMG_WIDTH
    test_ds = MultiFrameDataset(
        root_dir=test_dir,
        mode='val',
        split_ratio=config.SPLIT_RATIO,
        img_height=in_h,
        img_width=in_w,
        char2idx=config.CHAR2IDX,
        is_test=True
    )
    
    if len(test_ds) == 0:
        print(f"❌ ERROR: No track directories found in {test_dir}")
        sys.exit(1)
        
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == 'cuda')
    )
    print(f"📊 Loaded {len(test_ds)} test tracks ({len(test_loader)} batches)")

    # 3. Instantiate model
    if args.model == "end2end":
        from src.models.end2end_lpr import End2EndLPR
        model = End2EndLPR(
            num_classes=config.NUM_CLASSES,
            sr_num_blocks=8,
            ocr_transformer_layers=2,
            use_stn=config.USE_STN,
            pretrained=False,
        )
    elif args.model == "restran":
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            pretrained=False,
        )
    else:
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
        )

    # 4. Load trained weights
    weights_path = args.weights
    if not os.path.exists(weights_path):
        print(f"❌ ERROR: Model checkpoint not found at {weights_path}")
        sys.exit(1)
        
    print(f"📦 Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()

    # 5. Run prediction loop
    print("🔮 Predict license plates...")
    results = []
    with torch.no_grad():
        for images, _, _, _, track_ids in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            preds = model(images)
            decoded_list = decode_with_confidence(preds, config.IDX2CHAR)
            
            for i, (pred_text, conf) in enumerate(decoded_list):
                results.append((track_ids[i], pred_text, conf))

    # 6. Save submission output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    submission_lines = [f"{track_id},{pred_text};{conf:.4f}" for track_id, pred_text, conf in results]
    with open(args.output_file, 'w') as f:
        f.write("\n".join(submission_lines))

    print(f"\n✅ SUBMISSION FILE CREATED SUCCESSFULLY: {args.output_file}")
    print(f"📈 Total Predictions Generated: {len(results)}")
    
    print("\n🔍 SAMPLE PREDICTIONS (First 15 Tracks):")
    print("-" * 50)
    for track_id, pred_text, conf in results[:15]:
        print(f"  📌 Track ID: {track_id:<25} | Plate: {pred_text:<12} | Conf: {conf:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    main()
