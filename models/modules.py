from typing import Sequence
import torch
import torch.nn as nn


class MultiScaleFusion(nn.Module):
    def __init__(self,
                 channels: Sequence[int] = (64, 128, 256)):
        super().__init__()
        
        self.l2_to_l1 = UpsampleConv(channels[1], channels[0], scale_factor=2)
        self.l3_to_l1 = UpsampleConv(channels[2], channels[0], scale_factor=4)
        
        # when groups == in_channels, means depthwise conv
        self.l1_to_l2 = nn.Conv2d(channels[0],  channels[1], stride=2, kernel_size=3, padding=1, groups=channels[0])
        self.l3_to_l2 = UpsampleConv(channels[2], channels[1], scale_factor=2)
        
        self.l1_to_l3 = nn.Conv2d(channels[0], channels[2], stride=4, kernel_size=5, padding=2, groups=channels[0])
        self.l2_to_l3 = nn.Conv2d(channels[1], channels[2], stride=2, kernel_size=3, padding=1, groups=channels[1])

    def forward(self, layer1_x, layer2_x, layer3_x):
        layer2_x_to_1 = self.l2_to_l1(layer2_x)
        layer3_x_to_1 = self.l3_to_l1(layer3_x)
        out1 = layer1_x + layer2_x_to_1 + layer3_x_to_1
        
        layer1_x_to_2 = self.l1_to_l2(layer1_x)
        layer3_x_to_2 = self.l3_to_l2(layer3_x)
        out2 = layer2_x + layer1_x_to_2 + layer3_x_to_2
        
        layer1_x_to_3 = self.l1_to_l3(layer1_x)
        layer2_x_to_3 = self.l2_to_l3(layer2_x)
        out3 = layer3_x + layer1_x_to_3 + layer2_x_to_3
        
        return out1, out2, out3
        
        
class UpsampleConv(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        
        return x


        