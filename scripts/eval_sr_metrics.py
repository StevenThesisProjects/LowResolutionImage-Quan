import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import get_default_config
from src.models.sr import StackedSRNet
from src.utils.common import seed_everything
from train_sr import SRDataset, denorm_to_uint8


@torch.no_grad()
def evaluate_sr_metrics(model, dataloader, device):
    """Evaluate both Bicubic baseline and StackedSRNet on paired Ground-Truth HR validation set."""
    model.eval()
    total_psnr_sr = 0.0
    total_ssim_sr = 0.0
    total_psnr_bic = 0.0
    total_ssim_bic = 0.0
    count = 0

    print("🔍 Evaluating PSNR & SSIM on Paired Ground-Truth HR Validation Set...")
    for lr, hr in tqdm(dataloader, desc="Eval SR"):
        lr, hr = lr.to(device), hr.to(device)
        if lr.dim() == 5:
            lr = lr[:, lr.size(1) // 2, :, :, :]

        # Bicubic baseline upsampling (2x)
        bic = torch.clamp(
            F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False),
            -1.0,
            1.0,
        )

        # StackedSRNet prediction
        sr = model(lr)

        B = lr.size(0)
        for i in range(B):
            sr_np = denorm_to_uint8(sr[i])
            bic_np = denorm_to_uint8(bic[i])
            hr_np = denorm_to_uint8(hr[i])

            total_psnr_sr += psnr(hr_np, sr_np, data_range=255)
            total_ssim_sr += ssim(hr_np, sr_np, channel_axis=-1, data_range=255, win_size=7)

            total_psnr_bic += psnr(hr_np, bic_np, data_range=255)
            total_ssim_bic += ssim(hr_np, bic_np, channel_axis=-1, data_range=255, win_size=7)
            count += 1

    if count == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        total_psnr_sr / count,
        total_ssim_sr / count,
        total_psnr_bic / count,
        total_ssim_bic / count,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate SR PSNR & SSIM on Paired Ground-Truth Validation Set")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to SR checkpoint")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for quick test")
    args = parser.parse_args()

    config = get_default_config()
    device = config.DEVICE
    seed_everything(config.SEED)

    val_dataset = SRDataset(
        data_root=config.DATA_ROOT,
        split_file=config.VAL_SPLIT_FILE,
        mode="val",
        img_height=config.IMG_HEIGHT // 2,
        img_width=config.IMG_WIDTH // 2,
        scale_factor=2,
        max_samples=args.max_samples,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if device.type == "mps" else 2,
    )

    model = StackedSRNet(num_blocks=8, scale_factor=2).to(device)
    sr_weights_path = args.checkpoint or os.path.join(config.OUTPUT_DIR, "sr_net_best.pth")

    if os.path.exists(sr_weights_path):
        print(f"📦 Loading SR weights from {sr_weights_path}")
        state = torch.load(sr_weights_path, map_location=device)
        if any(k.startswith("sr_net.") for k in state.keys()):
            state = {k.replace("sr_net.", ""): v for k, v in state.items() if k.startswith("sr_net.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"⚠️ Warning: Missing keys when loading SR checkpoint: {missing}")
    else:
        print(f"⚠️ Warning: {sr_weights_path} not found. Evaluating with random weights.")

    psnr_sr, ssim_sr, psnr_bic, ssim_bic = evaluate_sr_metrics(model, val_loader, device)

    print("\n" + "=" * 60)
    print("📊 SUPER-RESOLUTION METRICS (Paired Ground-Truth HR Val Set)")
    print("=" * 60)
    print(f"   - Bicubic Baseline  : PSNR = {psnr_bic:.2f} dB | SSIM = {ssim_bic:.4f}")
    print(f"   - StackedSRNet (SR) : PSNR = {psnr_sr:.2f} dB | SSIM = {ssim_sr:.4f}")
    print(f"   - Net Improvement   : ΔPSNR = {psnr_sr - psnr_bic:+.2f} dB | ΔSSIM = {ssim_sr - ssim_bic:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
