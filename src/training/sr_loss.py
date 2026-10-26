import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[0, 5, 10, 19, 28]):
        super(PerceptualLoss, self).__init__()
        # Use VGG19 features (pretrained on ImageNet)
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList()
        last_layer = 0
        for layer in feature_layers:
            self.blocks.append(vgg[last_layer:layer+1])
            last_layer = layer + 1
        
        # Freeze VGG parameters
        for p in self.parameters():
            p.requires_grad = False
            
        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        # Denormalize from [-1, 1] to [0, 1]
        sr_01 = torch.clamp((sr + 1.0) / 2.0, 0.0, 1.0)
        hr_01 = torch.clamp((hr + 1.0) / 2.0, 0.0, 1.0)
        
        # Normalize assuming inputs are in [0, 1]
        sr_norm = (sr_01 - self.mean) / self.std
        hr_norm = (hr_01 - self.mean) / self.std
        
        loss = 0.0
        x_sr = sr_norm
        x_hr = hr_norm
        for block in self.blocks:
            x_sr = block(x_sr)
            x_hr = block(x_hr)
            loss += F.l1_loss(x_sr, x_hr)
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Laplacian filter
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer('weight', kernel)

    def forward(self, sr, hr):
        # Apply Laplacian filter (channel-wise)
        sr_edge = F.conv2d(sr, self.weight, padding=1, groups=3)
        hr_edge = F.conv2d(hr, self.weight, padding=1, groups=3)
        return F.l1_loss(sr_edge, hr_edge)

class SRLoss(nn.Module):
    """
    Combined Loss for Super Resolution.
    L = w0 * MSE + w1 * Perceptual + w2 * Edge
    """
    def __init__(self, w_mse=1.0, w_perceptual=0.1, w_edge=0.1, device='cpu'):
        super(SRLoss, self).__init__()
        self.w_mse = w_mse
        self.w_perceptual = w_perceptual
        self.w_edge = w_edge
        
        self.mse = nn.MSELoss()
        
        if self.w_perceptual > 0:
            self.perceptual = PerceptualLoss().to(device)
            
        if self.w_edge > 0:
            self.edge = EdgeLoss().to(device)

    def forward(self, sr, hr):
        loss_mse = self.mse(sr, hr)
        loss = self.w_mse * loss_mse
        
        if self.w_perceptual > 0:
            loss_perc = self.perceptual(sr, hr)
            loss += self.w_perceptual * loss_perc
            
        if self.w_edge > 0:
            loss_edge = self.edge(sr, hr)
            loss += self.w_edge * loss_edge
            
        return loss
