from typing import List
import torch
from torch import Tensor
import torch.nn as nn


class MultiSizeAttentionModule(nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 heights: List[int],
                 num_layers: int = 3):
        super().__init__()
        
        layer1_msa, layer2_msa, layer3_msa = [], [], []
        for _ in range(num_layers):
            layer1_msa.append(MultiSizeAttention(in_channels[0], heights[0]))
            layer2_msa.append(MultiSizeAttention(in_channels[1], heights[1]))
            layer3_msa.append(MultiSizeAttention(in_channels[2], heights[2]))
        self.layer1_msa = nn.Sequential(*layer1_msa)
        self.layer2_msa = nn.Sequential(*layer2_msa)
        self.layer3_msa = nn.Sequential(*layer3_msa)
    
    def forward(self, features: List[Tensor]) -> List[Tensor]:
        out1 = self.layer1_msa(features[0])
        out2 = self.layer2_msa(features[1])
        out3 = self.layer3_msa(features[2])
        
        return out1, out2, out3
        
        
class MultiSizeAttention(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 height: int):
        super().__init__()
        
        self.ps1 = height
        proj_channels1 = in_channels * self.ps1 * self.ps1
        # the projection channels is too large, the only way we can do projection is by depthwise
        self.q_proj1 = nn.Conv1d(proj_channels1, proj_channels1, kernel_size=1, groups=proj_channels1)
        self.k_proj1 = nn.Conv1d(proj_channels1, proj_channels1, kernel_size=1, groups=proj_channels1)
        self.v_proj1 = nn.Conv1d(proj_channels1, proj_channels1, kernel_size=1, groups=proj_channels1)
        self.scale_factor1 = 1 / proj_channels1
        
        self.ps2 = height // 2
        proj_channels2 = in_channels * self.ps2 * self.ps2
        self.q_proj2 = nn.Conv1d(proj_channels2, proj_channels2, kernel_size=1, groups=proj_channels2)
        self.k_proj2 = nn.Conv1d(proj_channels2, proj_channels2, kernel_size=1, groups=proj_channels2)
        self.v_proj2 = nn.Conv1d(proj_channels2, proj_channels2, kernel_size=1, groups=proj_channels2)
        self.scale_factor2 = 1 / proj_channels2
        
        self.ps3 = height // 4
        proj_channels3 = in_channels * self.ps3 * self.ps3
        self.q_proj3 = nn.Conv1d(proj_channels3, proj_channels3, kernel_size=1, groups=proj_channels3)
        self.k_proj3 = nn.Conv1d(proj_channels3, proj_channels3, kernel_size=1, groups=proj_channels3)
        self.v_proj3 = nn.Conv1d(proj_channels3, proj_channels3, kernel_size=1, groups=proj_channels3)
        self.scale_factor3 = 1 / proj_channels3
        
        self.ps4 = height // 8
        proj_channels4 = in_channels * self.ps4 * self.ps4
        self.q_proj4 = nn.Conv1d(proj_channels4, proj_channels4, kernel_size=1, groups=proj_channels4)
        self.k_proj4 = nn.Conv1d(proj_channels4, proj_channels4, kernel_size=1, groups=proj_channels4)
        self.v_proj4 = nn.Conv1d(proj_channels4, proj_channels4, kernel_size=1, groups=proj_channels4)
        self.scale_factor4 = 1 / proj_channels4
        
        self.residual_block = ResidualBlock(4 * in_channels, in_channels)
        
    def forward(self, x: Tensor):
        B, C, H, W = x.shape

        x1 = self.patchify(x, self.ps1, C)  # (B, L, dim)
        x2 = self.patchify(x, self.ps2, C)
        x3 = self.patchify(x, self.ps3, C)
        x4 = self.patchify(x, self.ps4, C)
        x1 = x1.permute(0, 2, 1).contiguous()  # (B, dim, L)
        x2 = x2.permute(0, 2, 1).contiguous()
        x3 = x3.permute(0, 2, 1).contiguous()
        x4 = x4.permute(0, 2, 1).contiguous()
        
        q1 = self.q_proj1(x1).permute(0, 2, 1)  # (B, L, dim)
        k1 = self.k_proj1(x1).permute(0, 2, 1)
        v1 = self.v_proj1(x1).permute(0, 2, 1)
        
        q2 = self.q_proj2(x2).permute(0, 2, 1)
        k2 = self.k_proj2(x2).permute(0, 2, 1)
        v2 = self.v_proj2(x2).permute(0, 2, 1)
        
        q3 = self.q_proj3(x3).permute(0, 2, 1)
        k3 = self.k_proj3(x3).permute(0, 2, 1)
        v3 = self.v_proj3(x3).permute(0, 2, 1)

        q4 = self.q_proj4(x4).permute(0, 2, 1)
        k4 = self.k_proj4(x4).permute(0, 2, 1)
        v4 = self.v_proj4(x4).permute(0, 2, 1)
        
        attn = torch.einsum("ble,bse->bls", q1, k1)
        softmax_attn = torch.softmax(attn * self.scale_factor1, dim=-1)
        out1 = torch.einsum("bls,bsd->bld", softmax_attn, v1)
        out1 = self.unpatchify(out1, self.ps1, C)
        
        attn = torch.einsum("ble,bse->bls", q2, k2)
        softmax_attn = torch.softmax(attn * self.scale_factor2, dim=-1)
        out2 = torch.einsum("bls,bsd->bld", softmax_attn, v2)
        out2 = self.unpatchify(out2, self.ps2, C)
        
        attn = torch.einsum("ble,bse->bls", q3, k3)
        softmax_attn = torch.softmax(attn * self.scale_factor3, dim=-1)
        out3 = torch.einsum("bls,bsd->bld", softmax_attn, v3)
        out3 = self.unpatchify(out3, self.ps3, C)
        
        attn = torch.einsum("ble,bse->bls", q4, k4)
        softmax_attn = torch.softmax(attn * self.scale_factor4, dim=-1)
        out4 = torch.einsum("bls,bsd->bld", softmax_attn, v4)
        out4 = self.unpatchify(out4, self.ps4, C)
        
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.residual_block(out)
        
        return out
    
    def patchify(self, x, patch_size=16, num_channels=128):
        """
        Convert a feature map into patches of features.
        
        Args:
            x: shape of (B, C, H, W).
        Returns:
            patches: (B, L, num_channels * patch_size**2).
        """
        assert x.shape[2] == x.shape[3] and x.shape[2] % patch_size == 0

        h = w = x.shape[2] // patch_size
        patches = x.reshape(shape=(x.shape[0], num_channels, h, patch_size, w, patch_size))
        patches = torch.einsum('nchpwq->nhwcpq', patches)
        patches = patches.reshape(shape=(x.shape[0], h * w, num_channels * patch_size**2))
        
        return patches

    def unpatchify(self, x, patch_size=16, num_channels=128):
        """
            x: (N, L, num_channels * patch_size**2)
            out: (N, num_channels, H, W)
        """
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, num_channels, patch_size, patch_size))
        x = torch.einsum('nhwcpq->nchpwq', x)
        out = x.reshape(shape=(x.shape[0], num_channels, h * patch_size, w * patch_size))
        
        return out


class ResidualBlock(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(in_channels, channels, 1, stride=1, padding=0)

    def forward(self, x):
        """Forward function."""
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = self.relu(out)

        return out
