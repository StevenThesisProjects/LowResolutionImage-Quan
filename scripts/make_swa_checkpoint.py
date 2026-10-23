"""Stochastic Weight Averaging (SWA / Model Soup) for Flow 2 MultiFrameSRResTran."""
import os
from collections import OrderedDict
import torch


def main():
    checkpoints = [
        ("results/sr_restran_flow2_v22_best.pth", 0.50),
        ("results/sr_restran_flow2_plus10_best.pth", 0.30),
        ("results/sr_restran_flow2_push80_best.pth", 0.20),
    ]
    output_path = "results/sr_restran_flow2_swa_best.pth"

    print("🍲 Building Flow 2 SWA / Model Soup Checkpoint...")
    loaded_states = []
    for path, weight in checkpoints:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"  • Loading {path} (weight: {weight:.2f})")
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        loaded_states.append((state, weight))

    base_state, _ = loaded_states[0]
    swa_state = OrderedDict()

    for k in base_state.keys():
        t0 = base_state[k]
        if torch.is_floating_point(t0):
            acc = torch.zeros_like(t0, dtype=torch.float32)
            for state, weight in loaded_states:
                acc += weight * state[k].to(torch.float32)
            swa_state[k] = acc.to(t0.dtype)
        else:
            swa_state[k] = t0.clone()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(swa_state, output_path)
    print(f"✅ SWA Checkpoint saved successfully to: {output_path}")
    print(f"   Total keys averaged: {len(swa_state)}")


if __name__ == "__main__":
    main()
