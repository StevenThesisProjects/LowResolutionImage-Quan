import torch
import torch.nn as nn
import torch.nn.functional as F

class CALayer(nn.Module):
    """Channel Attention (CA) Layer from RCAN."""
    def __init__(self, channel: int, reduction: int = 16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) - EDSR style without BN + CALayer."""
    def __init__(self, in_feat: int, out_feat: int, reduction: int = 16):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_feat, out_feat, 3, padding=1, bias=True)
        self.ca = CALayer(out_feat, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.ca(res)
        res += x
        return res

class StackedSRNet(nn.Module):
    """
    Frame-Parallel Super-Resolution Network.
    Input: [Batch, Frames, 3, H, W] -> Output: [Batch, Frames, 3, H*scale, W*scale]
    Or Input: [Batch*Frames, 3, H, W] -> Output: [Batch*Frames, 3, H*scale, W*scale]
    """
    def __init__(
        self,
        num_frames: int = 5,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 8,
        scale_factor: int = 2
    ):
        super(StackedSRNet, self).__init__()
        self.num_frames = num_frames
        self.scale_factor = scale_factor
        
        # Initial feature extraction (no more 15-channel stacking)
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=True)
        
        # Residual body with RCAB blocks
        self.body = nn.Sequential(*[
            RCAB(base_channels, base_channels, reduction=16) for _ in range(num_blocks)
        ])
        
        self.conv_body = nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=True)
        
        # Upsampling (PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * (scale_factor ** 2), 3, padding=1, bias=True),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(inplace=True)
        )
        
        # Final reconstruction layer
        self.tail = nn.Conv2d(base_channels, 3, 3, padding=1, bias=True)
        # Learnable residual scaling gate (1.0 for pretrained SR, 0.05 for random-SR warmstart)
        self.res_scale_gate = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, C, H, W] or [B*N, C, H, W].
        Returns:
            HR image tensor of shape [B, N, C, H*scale, W*scale] or [B*N, C, H*scale, W*scale].
        """
        is_5d = x.dim() == 5
        if is_5d:
            B, N, C, H, W = x.size()
            x = x.view(B * N, C, H, W)
            
        # Global residual connection via bicubic interpolation
        base = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        
        feat = self.head(x)
        res = self.body(feat)
        res = self.conv_body(res)
        res += feat
        
        out = self.tail(self.upsample(res))
        
        # Add gated SR residual to bicubic base and clamp to [-1, 1]
        out = torch.clamp(base + self.res_scale_gate * out, -1.0, 1.0)
        
        if is_5d:
            _, C_out, H_out, W_out = out.size()
            out = out.view(B, N, C_out, H_out, W_out)
            
        return out
