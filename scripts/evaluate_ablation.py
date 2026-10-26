import torch
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.end2end_lpr import End2EndLPR
from configs.config import get_default_config

def measure_latency(model, input_tensor, device, num_runs=100):
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)
            
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(input_tensor)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # ms
            
    latencies.sort()
    p95_latency = latencies[int(0.95 * len(latencies))]
    return p95_latency

def main():
    print("=== Task 4.1 & 4.3: Ablation & Resource Benchmark ===")
    config = get_default_config()
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    print(f"Running on Device: {device}")
    
    from src.models.restran import ResTranOCR
    from src.models.sr_restran import MultiFrameSRResTran
    from src.models.sr_vit import MultiFrameSRViT
    from src.models.crnn import MultiFrameCRNN
    
    # Define Architectures
    architectures = [
        {"name": "Flow 1: CRNN (+STN)", "type": "crnn", "use_stn": True},
        {"name": "Flow 1: ResTran34 Baseline (+STN)", "type": "restran", "use_stn": True},
        {"name": "Flow 1: ResTran34 (No STN)", "type": "restran", "use_stn": False},
        {"name": "Flow 2: MultiFrameSRResTran (+STN)", "type": "sr_restran", "use_stn": True},
        {"name": "Flow 2: MultiFrameSRResTran (No STN)", "type": "sr_restran", "use_stn": False},
        {"name": "Flow 3: MultiFrameSRViT (Proposed)", "type": "sr_vit", "use_stn": False},
    ]
    
    lr_input = torch.randn(1, 5, 3, 16, 64).to(device)
    hr_input = torch.randn(1, 5, 3, 32, 128).to(device)
    
    print(f"{'Architecture / Pipeline':<40} | {'Params (M)':<12} | {'P95 Latency (ms)':<16} | {'FPS':<10}")
    print("-" * 85)
    
    for arch in architectures:
        a_type = arch["type"]
        if a_type == "crnn":
            model = MultiFrameCRNN(num_classes=config.NUM_CLASSES, use_stn=arch["use_stn"]).to(device)
            inp = hr_input
        elif a_type == "restran":
            model = ResTranOCR(num_classes=config.NUM_CLASSES, use_stn=arch["use_stn"], pretrained=False).to(device)
            inp = hr_input
        elif a_type == "sr_restran":
            model = MultiFrameSRResTran(num_classes=config.NUM_CLASSES, use_stn=arch["use_stn"], pretrained=False).to(device)
            inp = lr_input
        elif a_type == "sr_vit":
            model = MultiFrameSRViT(num_classes=config.NUM_CLASSES).to(device)
            inp = lr_input
            
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        latency = measure_latency(model, inp, device)
        fps = 1000.0 / latency if latency > 0 else 0
        
        print(f"{arch['name']:<40} | {total_params:>10.2f} M | {latency:>10.2f} ms | {fps:>8.1f}")

if __name__ == "__main__":
    main()
