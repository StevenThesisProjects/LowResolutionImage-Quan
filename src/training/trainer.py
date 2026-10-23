"""Trainer class encapsulating the training and validation loop."""
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence
from src.training.ctc_loss import HardCharacterCTCLoss


class Trainer:
    """Encapsulates training, validation, and inference logic."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str]
    ):
        """
        Args:
            model: The neural network model.
            train_loader: Training data loader.
            val_loader: Validation data loader (can be None).
            config: Configuration object with training parameters.
            idx2char: Index to character mapping for decoding.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)
        
        # Loss and optimizer
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        self.use_hard_loss: bool = getattr(config, 'USE_HARD_LOSS', False)
        self.hard_criterion = HardCharacterCTCLoss(
            char2idx=getattr(config, 'CHAR2IDX', {}),
            blank=0,
            alpha_hard=0.5,
            gamma_focal=0.5,
            margin=0.10,
            lambda_margin=2.0,
        )

        # Build differential parameter groups (filtering out any empty groups for --freeze-sr safety)
        base_lr = config.LEARNING_RATE
        warmstart_ocr = getattr(config, 'WARMSTART_OCR', False)

        stn_params: List[nn.Parameter] = []
        sr_params: List[nn.Parameter] = []
        ocr_params: List[nn.Parameter] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'stn.' in name:
                stn_params.append(param)
            elif name.startswith('sr_net.'):
                sr_params.append(param)
            else:
                ocr_params.append(param)

        self.stn_params = stn_params
        has_checkpoint = getattr(config, 'HAS_CHECKPOINT', False)
        warmstart_sr = getattr(config, 'WARMSTART_SR', False)

        param_groups = []
        max_lrs = []
        if sr_params:
            if has_checkpoint:
                sr_lr = base_lr * 0.2
            elif warmstart_sr:
                sr_lr = base_lr * 0.5
            else:
                sr_lr = base_lr
            param_groups.append({'params': sr_params, 'lr': sr_lr})
            max_lrs.append(sr_lr)
        if ocr_params:
            ocr_scale = getattr(config, 'OCR_LR_SCALE', 1.0)
            ocr_lr = (base_lr * 0.5 if (warmstart_ocr and hasattr(model, 'sr_net')) else base_lr) * ocr_scale
            param_groups.append({'params': ocr_params, 'lr': ocr_lr})
            max_lrs.append(ocr_lr)
        if stn_params:
            stn_lr = base_lr * 0.1
            param_groups.append({'params': stn_params, 'lr': stn_lr})
            max_lrs.append(stn_lr)

        if not param_groups:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            param_groups = [{'params': trainable_params, 'lr': base_lr}]
            max_lrs = [base_lr]

        self.optimizer = optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=config.WEIGHT_DECAY
        )
        total_steps = max(1, len(train_loader) * config.EPOCHS)
        pct_start = max(0.05 if has_checkpoint else 0.10, 2.0 / max(4, total_steps))
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lrs if len(max_lrs) > 1 else max_lrs[0],
            steps_per_epoch=max(1, len(train_loader)),
            epochs=config.EPOCHS,
            pct_start=pct_start,
            div_factor=10.0 if has_checkpoint else 25.0,
            final_div_factor=100.0 if has_checkpoint else 10000.0,
        )
        self.device_type: str = self.device.type if self.device.type in ['cuda', 'mps', 'cpu'] else 'cpu'
        self.autocast_enabled: bool = (self.device.type in ['cuda', 'mps'])
        self.use_cuda_scaler: bool = (self.device.type == 'cuda')
        self.scaler: GradScaler = GradScaler('cuda', enabled=self.use_cuda_scaler)
        
        # Tracking
        self.best_acc = 0.0
        self.current_epoch = 0
    
    def _get_output_path(self, filename: str) -> str:
        """Get full path for output file in configured directory."""
        output_dir = getattr(self.config, 'OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    
    def _get_exp_name(self) -> str:
        """Get experiment name from config."""
        return getattr(self.config, 'EXPERIMENT_NAME', 'baseline')

    def _get_head_weight(self) -> Optional[torch.Tensor]:
        """Extract linear classification head weight matrix for Prototype Margin Regularization."""
        if hasattr(self.model, 'ocr_net') and hasattr(self.model.ocr_net, 'head'):
            return self.model.ocr_net.head.weight
        if hasattr(self.model, 'head') and isinstance(self.model.head, nn.Linear):
            return self.model.head.weight
        if hasattr(self.model, 'ctc_head') and isinstance(self.model.ctc_head, nn.Linear):
            return self.model.ctc_head.weight
        return None

    def train_one_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        if getattr(self.config, 'FREEZE_SR', False) and hasattr(self.model, 'sr_net'):
            self.model.sr_net.eval()
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")
        
        for images, targets, target_lengths, _, _ in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            sr_trainable = (
                hasattr(self.model, 'sr_net')
                and not getattr(self.config, 'FREEZE_SR', False)
                and any(p.requires_grad for n, p in self.model.sr_net.named_parameters() if n != 'res_scale_gate')
            )
            with autocast(device_type=self.device_type, enabled=self.autocast_enabled):
                if sr_trainable:
                    preds, sr_images = self.model(images, return_sr=True)
                else:
                    preds = self.model(images)
                    sr_images = None

                preds_permuted = preds.permute(1, 0, 2)
                # CRITICAL FIX (Bug 2): CTC loss computation in Float32 with clamping to avoid FP16 underflow -> NaN
                log_probs = torch.clamp(preds_permuted.float(), min=-50.0, max=0.0)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=preds.size(1),
                    dtype=torch.long
                )
                if self.use_hard_loss:
                    loss_ctc = self.hard_criterion(
                        log_probs,
                        targets,
                        input_lengths,
                        target_lengths,
                        head_weight=self._get_head_weight(),
                    )
                elif self.device.type == 'mps':
                    loss_ctc = self.criterion(
                        log_probs.cpu(),
                        targets.cpu(),
                        input_lengths,
                        target_lengths,
                    ).to(self.device)
                else:
                    loss_ctc = self.criterion(
                        log_probs,
                        targets,
                        input_lengths,
                        target_lengths,
                    )

                if sr_images is not None:
                    # Self-Consistent LR-Downprojection Supervision + Total Variation Regularization (Review 3 Bug #4 Fix)
                    b_sz, f_sz, c_sz, h_sr, w_sr = sr_images.shape
                    sr_flat = sr_images.float().view(b_sz * f_sz, c_sz, h_sr, w_sr)
                    lr_flat = images.float().view(b_sz * f_sz, c_sz, images.size(3), images.size(4))
                    sr_down = torch.nn.functional.interpolate(
                        sr_flat, size=(images.size(3), images.size(4)), mode='area'
                    )
                    loss_lr_consist = torch.nn.functional.l1_loss(sr_down, lr_flat)
                    loss_tv = (
                        torch.mean(torch.abs(sr_flat[:, :, :, 1:] - sr_flat[:, :, :, :-1]))
                        + torch.mean(torch.abs(sr_flat[:, :, 1:, :] - sr_flat[:, :, :-1, :]))
                    )
                    loss = loss_ctc + 0.20 * loss_lr_consist + 0.01 * loss_tv
                else:
                    loss = loss_ctc

            # NaN / Inf Loss Guard (Bug 2 Fix)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ WARNING: NaN/Inf loss ({loss.item()}) detected! Skipping batch backward step to protect optimizer states.")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Scale loss & backward
            self.scaler.scale(loss).backward()
            
            # Unscale (required before gradient clipping)
            self.scaler.unscale_(self.optimizer)
            
            # Dedicated STN gradient clipping to prevent Epoch-10/14 affine explosion
            if self.stn_params:
                torch.nn.utils.clip_grad_norm_(self.stn_params, max_norm=0.5)
            
            # Global gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            
            # Save scale before step for scheduler check
            scale_before = self.scaler.get_scale()
            
            # Step optimizer & update scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Step scheduler only if optimizer actually stepped (scale not reduced)
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]})

            if getattr(self.config, 'DRY_RUN', False) and pbar.n >= 5:
                print(f"\n   [DRY-RUN] Reached 5 batches in training loop.")
                break
        
        num_batches = pbar.n if (getattr(self.config, 'DRY_RUN', False) and pbar.n > 0) else len(self.train_loader)
        return epoch_loss / num_batches

    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        """Run validation and generate submission data.
        
        Returns:
            Tuple of (metrics_dict, submission_data).
            metrics_dict contains at least 'loss' and 'acc'.
        """
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0, 'cer': 0.0}, []
        
        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds: List[str] = []
        all_targets: List[str] = []
        submission_data: List[str] = []
        
        with torch.no_grad():
            for images, targets, target_lengths, labels_text, track_ids in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                with autocast(device_type=self.device_type, enabled=self.autocast_enabled):
                    preds = self.model(images)
                    preds_permuted = preds.permute(1, 0, 2)
                    # CRITICAL FIX (Bug 2): CTC loss computation in Float32 with clamping to avoid FP16 underflow -> NaN
                    log_probs = torch.clamp(preds_permuted.float(), min=-50.0, max=0.0)
                    input_lengths = torch.full(
                        (images.size(0),),
                        preds.size(1),
                        dtype=torch.long
                    )
                    if self.device.type == 'mps':
                        loss = self.criterion(
                            log_probs.cpu(),
                            targets.cpu(),
                            input_lengths,
                            target_lengths
                        )
                    else:
                        loss = self.criterion(
                            log_probs,
                            targets,
                            input_lengths,
                            target_lengths
                        )
                val_loss += loss.item()

                # Decode predictions
                decoded_list = decode_with_confidence(preds, self.idx2char)

                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    
                    all_preds.append(pred_text)
                    all_targets.append(gt_text)
                    
                    if pred_text == gt_text:
                        total_correct += 1
                    submission_data.append(f"{track_id},{pred_text};{conf:.4f}")
                    
                total_samples += len(labels_text)

                if getattr(self.config, 'DRY_RUN', False) and total_samples >= 32:
                    print(f"   [DRY-RUN] Reached sample validation limit ({total_samples} samples).")
                    break

        num_val_batches = len(self.val_loader) if not getattr(self.config, 'DRY_RUN', False) else max(1, total_samples // getattr(self.config, 'BATCH_SIZE', 32))
        avg_val_loss = val_loss / num_val_batches
        val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
        
        metrics = {
            'loss': avg_val_loss,
            'acc': val_acc,
        }
        
        return metrics, submission_data

    def save_submission(self, submission_data: List[str]) -> None:
        """Save submission file with experiment name."""
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"submission_{exp_name}.txt")
        with open(filename, 'w') as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {filename}")

    def save_model(self, path: str = None) -> None:
        """Save model checkpoint with experiment name."""
        if path is None:
            exp_name = self._get_exp_name()
            path = self._get_output_path(f"{exp_name}_best.pth")
        torch.save(self.model.state_dict(), path)

    def fit(self) -> None:
        """Run the full training loop for specified number of epochs."""
        print(f"🚀 TRAINING START | Device: {self.device} | Epochs: {self.config.EPOCHS}")
        start_time = time.time()
        max_hours = getattr(self.config, 'MAX_HOURS', None)
        if max_hours:
            print(f"🛡️ HARD TIME GUARD ENABLED: Training will automatically terminate after {max_hours:.2f} hours to protect GPU quota.")
        
        if self.val_loader is not None and (
            getattr(self.config, 'WARMSTART_OCR', False)
            or getattr(self.config, 'HAS_CHECKPOINT', False)
        ):
            init_metrics, init_sub = self.validate()
            self.best_acc = init_metrics['acc']
            print(f"📌 Epoch 0 (Warm-Start Baseline): Val Loss: {init_metrics['loss']:.4f} | Val Acc: {self.best_acc:.2f}%")
            if self.best_acc > 0:
                self.save_model()
                if init_sub:
                    self.save_submission(init_sub)

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            
            # Training
            avg_train_loss = self.train_one_epoch()
            
            # Validation
            val_metrics, submission_data = self.validate()
            val_loss = val_metrics['loss']
            val_acc = val_metrics['acc']
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log results
            print(f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.2e}")
            
            # Save best model
            exp_name = self._get_exp_name()
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_model()
                model_path = self._get_output_path(f"{exp_name}_best.pth")
                print(f"  ⭐ Saved Best Model: {model_path} ({val_acc:.2f}%)")
                
                if submission_data:
                    self.save_submission(submission_data)

            # Always save latest model checkpoint (Bug 3 fix)
            latest_path = self._get_output_path(f"{exp_name}_latest.pth")
            self.save_model(latest_path)
            
            # If dry-run, only run 1 epoch
            if getattr(self.config, 'DRY_RUN', False):
                print(f"   [DRY-RUN] Completed 1 dry-run epoch successfully.")
                break

            # Hard Max-Hours Time Guard (Bug 6 fix): Prevent runaway GPU jobs / hangs
            if max_hours is not None:
                elapsed_hours = (time.time() - start_time) / 3600.0
                if elapsed_hours >= max_hours:
                    print(f"\n⏰ TIME GUARD LIMIT REACHED: Training elapsed {elapsed_hours:.2f}h >= {max_hours:.2f}h.")
                    print(f"   Terminating training safely to conserve GPU quota. Checkpoint safely saved at epoch {epoch + 1}.")
                    break
        
        # Save final model if no validation was performed (submission mode)
        exp_name = self._get_exp_name()
        if self.val_loader is None:
            self.save_model()
            model_path = self._get_output_path(f"{exp_name}_best.pth")
            print(f"  💾 Saved final model: {model_path}")

        # Safety Fallback (Bug 3 Fix): Ensure best.pth always exists even if val_acc was 0.00%
        best_model_path = self._get_output_path(f"{exp_name}_best.pth")
        if not os.path.exists(best_model_path):
            self.save_model(best_model_path)
            print(f"  💾 Fallback: Initialized {best_model_path} from latest epoch weights.")
        
        print(f"\n✅ Training complete! Best Val Acc: {self.best_acc:.2f}%")

    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        """Run inference on a data loader.
        
        Returns:
            List of (track_id, predicted_text, confidence) tuples.
        """
        self.model.eval()
        results: List[Tuple[str, str, float]] = []
        
        with torch.no_grad():
            for images, _, _, _, track_ids in loader:
                images = images.to(self.device)
                with autocast(device_type=self.device_type, enabled=self.autocast_enabled):
                    preds = self.model(images)
                
                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
        
        return results

    def predict_test(self, test_loader: DataLoader, output_filename: str = "submission_final.txt") -> None:
        """Run inference on test data and save submission file.
        
        Args:
            test_loader: DataLoader for test data.
            output_filename: Name of the submission file to save.
        """
        print(f"🔮 Running inference on test data...")
        
        # Use existing predict method
        results = []
        self.model.eval()
        with torch.no_grad():
            for images, _, _, _, track_ids in tqdm(test_loader, desc="Test Inference"):
                images = images.to(self.device)
                with autocast(device_type=self.device_type, enabled=self.autocast_enabled):
                    preds = self.model(images)
                decoded_list = decode_with_confidence(preds, self.idx2char)
                
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
        
        # Format and save submission file
        submission_data = [f"{track_id},{pred_text};{conf:.4f}" for track_id, pred_text, conf in results]
        output_path = self._get_output_path(output_filename)
        with open(output_path, 'w') as f:
            f.write("\n".join(submission_data))
        
        print(f"✅ Saved {len(submission_data)} predictions to {output_path}")
