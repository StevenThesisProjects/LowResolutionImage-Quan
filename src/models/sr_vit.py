"""Flow 3: MultiFrameSRViT with Frame-Parallel SR, STN, Hybrid ResNet-ViT Tokenizer, and Spatio-Temporal Cross-Attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34

from src.models.components import STNBlock
from src.models.sr import StackedSRNet


class MultiFrameCrossAttentionFusion(nn.Module):
    """
    Spatio-Temporal Cross-Frame Attention over 1D horizontal sequence tokens.
    Center frame acts as Query [B, W', C]; all 5 frames act as Key/Value [B, F*W', C].
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: torch.Tensor, num_frames: int = 5) -> torch.Tensor:
        # features: [B*F, N, C]
        B_F, N, C = features.size()
        B = B_F // num_frames
        features_4d = features.reshape(B, num_frames, N, C)

        center_idx = num_frames // 2
        Q = features_4d[:, center_idx, :, :]              # [B, N, C]
        KV = features_4d.reshape(B, num_frames * N, C)    # [B, F*N, C]

        attn_out, _ = self.cross_attn(Q, KV, KV)
        return self.norm(Q + attn_out)                    # [B, N, C]


class MultiFrameSRViT(nn.Module):
    """
    Flow 3: Multi-Frame Super-Resolution + STN + Pretrained Hybrid ResNet-ViT Stem
    + Spatio-Temporal Cross-Attention + Pre-LN Vision Transformer Encoder.
    Total parameters: ~14.39M (55% lighter than MultiFrameSRResTran's 31.87M).
    """
    def __init__(
        self,
        num_classes: int,
        sr_num_blocks: int = 8,
        patch_h: int = 4,
        patch_w: int = 8,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        use_stn: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_stn = use_stn

        # 1. Frame-Parallel Super-Resolution (16x64 -> 32x128)
        self.sr_net = StackedSRNet(
            num_frames=5,
            in_channels=3,
            base_channels=64,
            num_blocks=sr_num_blocks,
            scale_factor=2,
        )

        # 2. Spatial Transformer Network on SR frames
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 3. Hybrid ResNet-ViT Stem (Early stages of ResNet-34, dropping the 13.1M-param layer4)
        # Input: [B*F, 3, 32, 128] -> layer3 output: [B*F, 256, 2, 16] -> Patch Proj: [B*F, embed_dim, 1, 16]
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        resnet.layer3[0].conv1.stride = (2, 1)
        resnet.layer3[0].downsample[0].stride = (2, 1)
        self.conv_stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            nn.Conv2d(256, embed_dim, kernel_size=(2, 1), stride=(2, 1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # 4. Learnable 1D Horizontal Positional & Temporal Frame Embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.frame_embed = nn.Parameter(torch.randn(1, 5, 1, embed_dim) * 0.02)

        # 5. Multi-Frame Spatio-Temporal Cross-Attention Fusion
        self.cross_attn = MultiFrameCrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 6. Pre-LN Vision Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 7. CTC Classification Head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, return_sr: bool = False):
        B, F_frames, C, H, W = x.size()

        # 1. Super-Resolution: [B, F, 3, H, W] -> [B, F, 3, 2H, 2W]
        sr_imgs = self.sr_net(x)
        _, _, _, sr_H, sr_W = sr_imgs.size()
        sr_flat = sr_imgs.view(B * F_frames, C, sr_H, sr_W)

        # 2. STN Alignment
        if self.use_stn:
            theta = self.stn(sr_flat)
            grid = F.affine_grid(theta, sr_flat.size(), align_corners=False)
            sr_aligned = F.grid_sample(sr_flat, grid, align_corners=False)
        else:
            sr_aligned = sr_flat

        # 3. Hybrid Conv Stem + Vertical Collapse -> [B*F, embed_dim, 1, W']
        feat = self.conv_stem(sr_aligned)
        feat = F.adaptive_avg_pool2d(feat, (1, None))
        seq_w = feat.size(3)

        # 4. Sequence Tokens [B*F, W', embed_dim] + Positional & Frame Embeddings
        tokens = feat.squeeze(2).permute(0, 2, 1)
        if seq_w <= self.pos_embed.size(1):
            pos = self.pos_embed[:, :seq_w, :]
        else:
            pos = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=seq_w,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
        tokens = tokens + pos

        tokens_4d = tokens.view(B, F_frames, seq_w, self.embed_dim)
        if F_frames <= self.frame_embed.size(1):
            tokens_4d = tokens_4d + self.frame_embed[:, :F_frames, :, :]
        tokens_3d = tokens_4d.view(B * F_frames, seq_w, self.embed_dim)

        # 5. Cross-Frame Attention Fusion -> [B, W', embed_dim]
        fused = self.cross_attn(tokens_3d, num_frames=F_frames)

        # 6. Pre-LN Transformer Encoder + FP32 Log-Softmax
        vit_out = self.norm(self.transformer(fused))
        logits = self.head(vit_out).float().log_softmax(2)

        if return_sr:
            return logits, sr_imgs
        return logits
