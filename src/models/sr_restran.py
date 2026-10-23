"""MultiFrameSRResTran: End-to-End Super-Resolution + ResTranOCR architecture (Flow 2 chuẩn)."""
import torch
import torch.nn as nn
from src.models.sr import StackedSRNet
from src.models.restran import ResTranOCR

class MultiFrameSRResTran(nn.Module):
    """
    Flow 2: Multi-Frame Super-Resolution (Frame-Parallel) + ResTranOCR (STN + ResNet-34 + AttentionFusion + Transformer).
    Pipeline: Input 5 LR frames [B, F, 3, H, W] -> StackedSRNet -> [B, F, 3, H*2, W*2] -> ResTranOCR -> CTC Logits [B, Seq_Len, Num_Classes].
    """
    def __init__(
        self,
        num_classes: int,
        sr_num_frames: int = 5,
        sr_in_channels: int = 3,
        sr_base_channels: int = 64,
        sr_num_blocks: int = 8,
        sr_scale_factor: int = 2,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        # 1. Super-Resolution Module (Frame-Parallel)
        self.sr_net = StackedSRNet(
            num_frames=sr_num_frames,
            in_channels=sr_in_channels,
            base_channels=sr_base_channels,
            num_blocks=sr_num_blocks,
            scale_factor=sr_scale_factor
        )
        # 2. OCR Module (ResTranOCR with STN, ResNet-34, AttentionFusion, Transformer)
        self.ocr_net = ResTranOCR(
            num_classes=num_classes,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            transformer_ff_dim=transformer_ff_dim,
            dropout=dropout,
            use_stn=use_stn,
            pretrained=pretrained
        )

    def forward(self, x: torch.Tensor, return_sr: bool = False):
        """
        Args:
            x: Input tensor [Batch, Frames, 3, H, W]
            return_sr: If True, returns (logits, sr_image) for loss computation
        Returns:
            logits: [Batch, Seq_Len, Num_Classes]
        """
        sr_conv_trainable = any(
            p.requires_grad for n, p in self.sr_net.named_parameters() if n != "res_scale_gate"
        )
        if sr_conv_trainable or self.sr_net.res_scale_gate.requires_grad:
            sr_image = self.sr_net(x)
        else:
            with torch.no_grad():
                sr_image = self.sr_net(x)
        logits = self.ocr_net(sr_image)
        if return_sr:
            return logits, sr_image
        return logits
