import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import get_default_config
from src.models.end2end_lpr import End2EndLPR
from src.models.sr_vit import MultiFrameSRViT

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_model(model, name, device, input_shape=(1, 5, 3, 16, 48), iterations=100, warmup=20):
    print(f"\n{'='*50}")
    print(f"🚀 Benchmarking {name} on {device}")
    print(f"{'='*50}")
    
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    print("🔥 Warming up...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            
    # Benchmark
    print(f"⏱️ Measuring P95 Latency over {iterations} iterations...")
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # ms
            
    latencies.sort()
    p95_latency = latencies[int(iterations * 0.95)]
    avg_latency = sum(latencies) / iterations
    fps = 1000.0 / avg_latency
    
    params = count_parameters(model)
    
    print(f"📊 Results for {name}:")
    print(f"   - Parameters : {params / 1e6:.2f} M")
    print(f"   - P95 Latency: {p95_latency:.2f} ms")
    print(f"   - Avg Latency: {avg_latency:.2f} ms")
    print(f"   - Throughput : {fps:.2f} FPS")

def main():
    config = get_default_config()
    device = config.DEVICE
    
    from src.models.sr_restran import MultiFrameSRResTran
    # Flow 2 Proposed Model
    flow2 = MultiFrameSRResTran(
        num_classes=config.NUM_CLASSES,
        sr_num_blocks=8,
        transformer_layers=config.TRANSFORMER_LAYERS,
        pretrained=False
    ).to(device)
    
    # Flow 3 Model
    flow3 = MultiFrameSRViT(
        num_classes=config.NUM_CLASSES,
        sr_num_blocks=8,
        patch_h=config.SR_VIT_PATCH_H,
        patch_w=config.SR_VIT_PATCH_W,
        embed_dim=config.SR_VIT_EMBED_DIM,
        num_heads=config.SR_VIT_NUM_HEADS,
        num_layers=config.SR_VIT_NUM_LAYERS,
        ff_dim=config.SR_VIT_FF_DIM,
        max_seq_len=(config.IMG_HEIGHT // config.SR_VIT_PATCH_H) * (config.IMG_WIDTH // config.SR_VIT_PATCH_W)
    ).to(device)
    
    # Run
    benchmark_model(flow2, "End2EndLPR (Flow 2)", device)
    benchmark_model(flow3, "MultiFrameSRViT (Flow 3)", device)

if __name__ == "__main__":
    main()
