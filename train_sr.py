import os
import time
import glob
import cv2
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.amp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
import albumentations.pytorch as Ap
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from configs.config import get_default_config
from src.models.sr import StackedSRNet
from src.training.sr_loss import SRLoss
from src.utils.common import seed_everything


def denorm_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor from [-1, 1] to uint8 numpy array [0, 255] in HWC format."""
    tensor = torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)
    arr = (tensor * 255.0).detach().cpu().numpy().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


class SRDataset(Dataset):
    """
    Dataset for Standalone Super-Resolution Warm-up (Stage 1a).
    Because StackedSRNet is Frame-Parallel (weights are 100% shared across frames with
    zero cross-frame parameters), each training step samples a random HR frame from the
    5-frame track (and the center HR frame in validation) and constructs a pixel-aligned
    Synthetic LR-HR pair:
    - Returns:
        lr_tensor: [3, lr_h, lr_w] in [-1, 1]
        hr_tensor: [3, hr_h, hr_w] in [-1, 1]
    """
    def __init__(
        self,
        data_root: str,
        split_file: str,
        mode: str = "train",
        img_height: int = 16,
        img_width: int = 64,
        scale_factor: int = 2,
        max_samples: int = None,
    ):
        self.data_root = data_root
        self.mode = mode
        self.img_height = img_height
        self.img_width = img_width
        self.scale_factor = scale_factor
        self.hr_height = img_height * scale_factor
        self.hr_width = img_width * scale_factor

        self.samples = []

        # Ensure valid split file exists using MultiFrameDataset safe split logic
        from src.data.dataset import MultiFrameDataset
        if not os.path.exists(split_file):
            os.makedirs(os.path.dirname(split_file), exist_ok=True)
            _ = MultiFrameDataset(
                root_dir=data_root,
                mode="train",
                val_split_file=split_file,
                split_ratio=0.9,
                seed=42,
            )

        with open(split_file, "r") as f:
            val_ids = set(json.load(f))

        abs_root = os.path.abspath(data_root)
        all_tracks = sorted(glob.glob(os.path.join(abs_root, "**", "track_*"), recursive=True))

        tracks = []
        for t in all_tracks:
            unique_id = MultiFrameDataset.get_track_unique_id(t)
            base_name = os.path.basename(t)
            is_val = unique_id in val_ids or (base_name in val_ids and "Scenario-B" in t)

            if mode == "train" and not is_val:
                tracks.append(t)
            elif mode == "val" and is_val:
                tracks.append(t)

        if mode == "train":
            random.Random(42).shuffle(tracks)

        if max_samples is not None and max_samples > 0:
            tracks = tracks[:max_samples]

        # Index tracks containing HR ground-truth frames
        for t in tracks:
            hr_files = sorted(
                glob.glob(os.path.join(t, "hr-*.png")) + glob.glob(os.path.join(t, "hr-*.jpg"))
            )
            if len(hr_files) > 0:
                self.samples.append({
                    "hr_paths": hr_files,
                    "center_hr_path": hr_files[len(hr_files) // 2],
                })

        # Mild realistic degradation applied to HR before downscaling to LR in train mode
        if self.mode == "train":
            self.degrade = A.Compose(
                [
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                            A.MotionBlur(blur_limit=(3, 3), p=1.0),
                        ],
                        p=0.4,
                    ),
                    A.ImageCompression(quality_range=(55, 90), p=0.4),
                ],
                is_check_shapes=False,
            )
        else:
            self.degrade = None

        # Consistent normalization in [-1, 1] matching StackedSRNet, SRLoss, and OCR pipeline
        self.lr_transform = A.Compose(
            [
                A.Resize(height=self.img_height, width=self.img_width, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                Ap.ToTensorV2(),
            ],
            is_check_shapes=False,
        )

        self.hr_transform = A.Compose(
            [
                A.Resize(height=self.hr_height, width=self.hr_width, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                Ap.ToTensorV2(),
            ],
            is_check_shapes=False,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        hr_paths = item["hr_paths"]

        # In train mode, sample any of the 5 HR frames for 5x diversity; in val mode, use center frame
        if self.mode == "train":
            chosen_path = random.choice(hr_paths)
        else:
            chosen_path = item["center_hr_path"]

        img = cv2.imread(chosen_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hr_img = cv2.resize(img, (self.hr_width, self.hr_height), interpolation=cv2.INTER_CUBIC)
        else:
            hr_img = np.zeros((self.hr_height, self.hr_width, 3), dtype=np.uint8)

        lr_source = hr_img.copy()
        if self.mode == "train" and self.degrade is not None and random.random() < 0.5:
            lr_source = self.degrade(image=lr_source)["image"]

        lr_tensor = self.lr_transform(image=lr_source)["image"]  # [3, 16, 64]
        hr_tensor = self.hr_transform(image=hr_img)["image"]     # [3, 32, 128]

        return lr_tensor, hr_tensor


@torch.no_grad()
def evaluate_sr(model, val_loader, criterion, device, scaler=None):
    """Evaluate SR model on validation set and return (val_loss, avg_psnr, avg_ssim)."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for lr, hr in tqdm(val_loader, desc="Validating SR", leave=False):
        lr, hr = lr.to(device), hr.to(device)
        if lr.dim() == 5:
            lr = lr[:, lr.size(1) // 2, :, :, :]

        with torch.amp.autocast(device_type=device.type) if scaler else torch.autocast("cpu", enabled=False):
            sr = model(lr)  # [B, 3, 32, 128]
            loss = criterion(sr.float(), hr.float())

        total_loss += loss.item()
        B = lr.size(0)
        for i in range(B):
            sr_np = denorm_to_uint8(sr[i])
            hr_np = denorm_to_uint8(hr[i])
            total_psnr += compute_psnr(hr_np, sr_np, data_range=255)
            total_ssim += compute_ssim(hr_np, sr_np, channel_axis=-1, data_range=255, win_size=7)
            count += 1

    avg_loss = total_loss / max(1, len(val_loader))
    avg_psnr = total_psnr / max(1, count)
    avg_ssim = total_ssim / max(1, count)
    return avg_loss, avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description="Standalone Super-Resolution Warm-up (Stage 1a)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of SR warmup epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (single frame per track)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (auto: 0 on MPS, 2 on CUDA)")
    parser.add_argument("--max-hours", type=float, default=1.5, help="Hard time guard in hours (Rule 10)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for quick local verification")
    parser.add_argument("--data-root", type=str, default=None, help="Override data root directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    config = get_default_config()
    seed_everything(config.SEED)

    data_root = args.data_root or config.DATA_ROOT
    output_dir = args.output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    device = config.DEVICE
    num_workers = args.num_workers if args.num_workers is not None else (0 if device.type == "mps" else 2)
    print("=" * 65)
    print("🚀 STAGE 1A: STANDALONE SUPER-RESOLUTION WARM-UP (Frame-Parallel)")
    print("=" * 65)
    print(f"   - Device       : {device}")
    print(f"   - Epochs       : {args.epochs}")
    print(f"   - Batch Size   : {args.batch_size}")
    print(f"   - Learning Rate: {args.lr}")
    print(f"   - Num Workers  : {num_workers}")
    print(f"   - Max Hours    : {args.max_hours}h (Rule 10 Quota Protection)")
    print("=" * 65)

    # Mock data directories if they don't exist for isolated unit test
    if not os.path.exists(data_root):
        print(f"⚠️ {data_root} not found. Creating dummy data for testing...")
        os.makedirs(os.path.join(data_root, "Scenario-B", "track_00001"), exist_ok=True)
        dummy_hr = np.zeros((32, 128, 3), dtype=np.uint8)
        for i in range(1, 6):
            cv2.imwrite(os.path.join(data_root, "Scenario-B", "track_00001", f"hr-00{i}.png"), dummy_hr)
            cv2.imwrite(os.path.join(data_root, "Scenario-B", "track_00001", f"lr-00{i}.png"), dummy_hr)
        with open(os.path.join(data_root, "Scenario-B", "track_00001", "annotations.json"), "w") as f:
            json.dump({"plate_text": "ABC1234"}, f)

    lr_h = config.IMG_HEIGHT // 2  # 16
    lr_w = config.IMG_WIDTH // 2   # 64

    train_dataset = SRDataset(
        data_root,
        config.VAL_SPLIT_FILE,
        mode="train",
        img_height=lr_h,
        img_width=lr_w,
        scale_factor=2,
        max_samples=args.max_samples,
    )
    val_dataset = SRDataset(
        data_root,
        config.VAL_SPLIT_FILE,
        mode="val",
        img_height=lr_h,
        img_width=lr_w,
        scale_factor=2,
    )

    print(f"📊 Dataset loaded: {len(train_dataset)} train tracks | {len(val_dataset)} val tracks")

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    model = StackedSRNet(
        num_frames=5,
        in_channels=3,
        base_channels=64,
        num_blocks=8,
        scale_factor=2,
    ).to(device)

    criterion = SRLoss(w_mse=1.0, w_perceptual=0.1, w_edge=0.1, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type) if device.type in ["cuda", "mps"] else None

    best_psnr = 0.0
    best_ssim = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        # Hard Time Guard check at start of epoch
        elapsed_hours = (time.time() - start_time) / 3600.0
        if args.max_hours is not None and elapsed_hours >= args.max_hours:
            print(f"\n⏰ [TIME GUARD] Reached {elapsed_hours:.2f}h >= {args.max_hours}h limit. Stopping safely.")
            break

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, (lr, hr) in enumerate(pbar):
            if step % 50 == 0 and args.max_hours is not None:
                if (time.time() - start_time) / 3600.0 >= args.max_hours:
                    print(f"\n⏰ [TIME GUARD] Mid-epoch limit ({args.max_hours}h) reached. Breaking safely.")
                    break

            lr, hr = lr.to(device), hr.to(device)
            if lr.dim() == 5:
                lr = lr[:, lr.size(1) // 2, :, :, :]
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type) if scaler else torch.autocast("cpu", enabled=False):
                sr = model(lr)  # [B, 3, 32, 128]
                loss = criterion(sr.float(), hr.float())

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        scheduler.step()
        avg_train_loss = train_loss / max(1, len(train_loader))

        val_loss, val_psnr, val_ssim = evaluate_sr(model, val_loader, criterion, device, scaler)
        print(
            f"📈 Epoch {epoch + 1:02d}/{args.epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}"
        )

        # Save latest checkpoint
        latest_path = os.path.join(output_dir, "sr_net_latest.pth")
        torch.save(model.state_dict(), latest_path)

        # Save best checkpoint by PSNR
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_path = os.path.join(output_dir, "sr_net_best.pth")
            warmup_path = os.path.join(output_dir, "sr_warmup_best.pth")
            torch.save(model.state_dict(), best_path)
            torch.save(model.state_dict(), warmup_path)
            print(f"  ⭐ Saved New Best SR Model -> {best_path} (PSNR: {best_psnr:.2f} dB | SSIM: {best_ssim:.4f})")

    total_mins = (time.time() - start_time) / 60.0
    print("=" * 65)
    print(f"✅ Standalone SR Warm-up Complete in {total_mins:.1f} mins!")
    print(f"🏆 Best Validation PSNR: {best_psnr:.2f} dB | Best SSIM: {best_ssim:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
