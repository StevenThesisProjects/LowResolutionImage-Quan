import torch
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import get_default_config
from train_joint import JointDataset
from src.models.sr_restran import MultiFrameSRResTran
from src.training.trainer import decode_with_confidence

def unnormalize(tensor):
    """Convert normalized tensor [3, H, W] to uint8 RGB image [H, W, 3]."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    # Undo Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    img = (img * 0.5 + 0.5) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def main():
    config = get_default_config()
    device = config.DEVICE
    print(f"Generating visual comparisons on Device: {device}...")
    
    val_dataset = JointDataset(config.DATA_ROOT, config.VAL_SPLIT_FILE, config.CHAR2IDX, mode='val')
    
    model = MultiFrameSRResTran(
        num_classes=config.NUM_CLASSES,
        sr_num_blocks=8,
        use_stn=True,
        pretrained=False
    ).to(device)
    
    checkpoint_path = "results/decoupled_restran_best.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "results/flow2_warmstart_init.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "results/sr_restran_best.pth"
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()
    
    os.makedirs("QQ/visualizations", exist_ok=True)
    
    # Pick a few diverse tracks
    indices = [0, 5, 10, 25, 50]
    
    with torch.no_grad():
        for count, idx in enumerate(indices):
            if idx >= len(val_dataset):
                continue
            item = val_dataset[idx]
            lr_tensor = item[0].unsqueeze(0).to(device) # [1, 5, 3, 16, 64]
            gt_text = item[4]
            track_id = item[5]
            
            logits, sr_tensor = model(lr_tensor, return_sr=True)
            decoded = decode_with_confidence(logits, config.IDX2CHAR)
            pred_text, conf = decoded[0]
            
            # Prepare Images
            if sr_tensor.dim() == 5:
                sr_img = unnormalize(sr_tensor[0, 2])
            else:
                sr_img = unnormalize(sr_tensor[0])
            lr_frames = [unnormalize(lr_tensor[0, f]) for f in range(5)]
            
            # Bicubic baseline for frame 2 (center)
            lr_center = lr_frames[2]
            bicubic_img = cv2.resize(lr_center, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            # Create a 2-row figure
            fig = plt.figure(figsize=(12, 5))
            
            # Row 1: 5 LR input frames
            for f in range(5):
                ax = plt.subplot(2, 5, f + 1)
                ax.imshow(lr_frames[f])
                ax.set_title(f"LR Frame {f+1}", fontsize=10)
                ax.axis('off')
                
            # Row 2: Bicubic vs StackedSRNet
            ax_bicubic = plt.subplot(2, 5, 7)
            ax_bicubic.imshow(bicubic_img)
            ax_bicubic.set_title("Bicubic Interpolation", fontsize=10)
            ax_bicubic.axis('off')
            
            ax_sr = plt.subplot(2, 5, 9)
            ax_sr.imshow(sr_img)
            ax_sr.set_title("Proposed StackedSRNet", fontsize=10, fontweight='bold', color='darkgreen')
            ax_sr.axis('off')
            
            # Overall Title
            status = "CORRECT" if pred_text == gt_text else "MISMATCH"
            color = "green" if pred_text == gt_text else "red"
            plt.suptitle(f"Track: {track_id} | Ground Truth: {gt_text} | Prediction: {pred_text} (Conf: {conf:.2f}) [{status}]",
                         fontsize=12, fontweight='bold', color=color, y=0.98)
            
            plt.tight_layout()
            out_file = f"QQ/visualizations/comparison_{count+1}_{track_id}.png"
            plt.savefig(out_file, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization to {out_file}")
            
    print("\n✅ Visual qualitative comparisons exported successfully to QQ/visualizations/!")

if __name__ == "__main__":
    main()
