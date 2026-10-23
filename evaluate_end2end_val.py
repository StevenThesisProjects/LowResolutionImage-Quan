import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from configs.config import get_default_config
from train_joint import JointDataset
from src.models.end2end_lpr import End2EndLPR
from src.training.trainer import decode_with_confidence
from scripts.eval_confusion import get_alignment

def main():
    config = get_default_config()
    device = config.DEVICE
    print(f"Evaluating on Device: {device}")
    
    val_dataset = JointDataset(config.DATA_ROOT, config.VAL_SPLIT_FILE, config.CHAR2IDX, mode='val')
    print(f"Loaded {len(val_dataset)} validation samples.")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=JointDataset.collate_fn)
    
    model = End2EndLPR(
        num_classes=config.NUM_CLASSES,
        sr_num_blocks=8,
        ocr_transformer_layers=2,
        pretrained=False
    ).to(device)
    checkpoint_path = "results/end2end_joint_best.pth"
    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    total_samples = 0
    exact_matches = 0
    total_edits = 0
    total_chars = 0
    conf_correct = []
    conf_wrong = []
    
    from collections import defaultdict
    confusions = defaultdict(int)
    
    with torch.no_grad():
        for lr, hr, targets, target_lengths, labels_text, track_ids, is_synth in tqdm(val_loader, desc="Validating End2End"):
            lr = lr.to(device)
            logits = model(lr, return_sr=False)
            
            decoded = decode_with_confidence(logits, config.IDX2CHAR)
            for i, (pred_text, conf) in enumerate(decoded):
                gt_text = labels_text[i]
                total_samples += 1
                total_chars += len(gt_text)
                
                if pred_text == gt_text:
                    exact_matches += 1
                    conf_correct.append(conf)
                else:
                    conf_wrong.append(conf)
                    
                ops = get_alignment(gt_text, pred_text)
                for op, r, h in ops:
                    if op == 'S':
                        confusions[(r, h)] += 1
                        total_edits += 1
                    elif op in ['I', 'D']:
                        total_edits += 1
                        
    acc = (exact_matches / total_samples) * 100
    cer = (total_edits / total_chars) * 100
    mean_conf_c = np.mean(conf_correct) if conf_correct else 0.0
    mean_conf_w = np.mean(conf_wrong) if conf_wrong else 0.0
    conf_gap = mean_conf_c - mean_conf_w
    
    print("\n" + "="*50)
    print("🎯 END-TO-END VALIDATION RESULTS (Real Data)")
    print("="*50)
    print(f"Exact Match Accuracy: {acc:.2f}%")
    print(f"Character Error Rate (CER): {cer:.2f}%")
    print(f"Confidence (Correct): {mean_conf_c:.4f}")
    print(f"Confidence (Wrong):   {mean_conf_w:.4f}")
    print(f"Confidence Gap:       +{conf_gap:.4f}")
    
    target_pairs = [('8', 'B'), ('B', '8'), ('0', 'D'), ('D', '0'), ('5', 'S'), ('S', '5'), ('2', '7'), ('7', '2')]
    print("\n--- Key Confusions on Hard Pairs ---")
    for (r, h) in target_pairs:
        if (r, h) in confusions:
            print(f"  {r} -> {h}: {confusions[(r, h)]} times")
            
    # Save results to json
    import json
    res = {
        'exact_match_acc': acc,
        'cer': cer,
        'conf_gap': conf_gap,
        'conf_correct': float(mean_conf_c),
        'conf_wrong': float(mean_conf_w)
    }
    with open("results/end2end_val_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\n✅ Metrics saved to results/end2end_val_metrics.json")

if __name__ == "__main__":
    main()
