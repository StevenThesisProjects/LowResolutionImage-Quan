import torch
import torch.nn as nn
from src.models.sr import StackedSRNet
from src.models.restran import ResTranOCR

class End2EndLPR(nn.Module):
    """
    End-to-End Low-Resolution License Plate Recognition Model.
    Pipeline: 5 LR Frames -> Frame-Parallel SRNet -> 5 Enhanced Frames -> STN + ResNet34 -> Attention Fusion -> Transformer -> CTC Head
    """
    def __init__(
        self,
        num_classes: int,
        # SR Configs
        sr_num_frames: int = 5,
        sr_in_channels: int = 3,
        sr_base_channels: int = 64,
        sr_num_blocks: int = 8,
        sr_scale_factor: int = 2,
        # OCR Configs
        ocr_transformer_heads: int = 8,
        ocr_transformer_layers: int = 3,
        ocr_transformer_ff_dim: int = 2048,
        ocr_dropout: float = 0.1,
        use_stn: bool = True,
        pretrained: bool = True,
    ):
        super(End2EndLPR, self).__init__()
        
        # Super-Resolution Module
        self.sr_net = StackedSRNet(
            num_frames=sr_num_frames,
            in_channels=sr_in_channels,
            base_channels=sr_base_channels,
            num_blocks=sr_num_blocks,
            scale_factor=sr_scale_factor
        )
        
        # OCR Module
        self.ocr_net = ResTranOCR(
            num_classes=num_classes,
            transformer_heads=ocr_transformer_heads,
            transformer_layers=ocr_transformer_layers,
            transformer_ff_dim=ocr_transformer_ff_dim,
            dropout=ocr_dropout,
            use_stn=use_stn,
            pretrained=pretrained
        )

    def forward(self, x: torch.Tensor, return_sr: bool = False):
        """
        Args:
            x: Input tensor of shape [Batch, Frames, 3, H, W]
            return_sr: If True, returns both the OCR logits and the intermediate SR image.
                       This is useful for computing the joint loss.
        Returns:
            If return_sr is True: (logits, sr_image)
            If return_sr is False: logits
        """
        # 1. Super Resolution: [B, F, C, H, W] -> [B, F, 3, H*scale, W*scale]
        sr_image = self.sr_net(x)
        
        # 2. OCR Pipeline (Receives full 5 frames -> AttentionFusion activated!)
        logits = self.ocr_net(sr_image)
        
        if return_sr:
            return logits, sr_image
        return logits
