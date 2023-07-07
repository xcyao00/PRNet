from typing import List
import timm
import torch
from torch import Tensor
import torch.nn as nn
from models.modules import MultiScaleFusion
from models.attention import MultiSizeAttentionModule
from utils import get_prototype_features, get_residual_features, get_concatenated_features


class PRNet(nn.Module):
    def __init__(self, 
                 backbone: str,
                 num_classes: int = 2,
                 input_size: tuple = (256, 256),
                 layers: tuple = (1, 2, 3, 4),
                 device: torch.device = torch.device('cuda')):
        super().__init__()
        
        self.backbone = backbone
       
        encoder = timm.create_model(backbone, features_only=True, 
            out_indices=layers, pretrained=True).eval()
        self.encoder = encoder.to(device)
        
        featuremap_dims = dryrun_find_featuremap_dims(self.encoder, input_size, len(layers), device)
        # Feature map height and width
        self.heights, self.widths, self.feature_dimensions = [], [], []
        for i in range(len(layers)):
            size = featuremap_dims[i]['resolution']
            self.heights.append(size[0])
            self.widths.append(size[1])
            self.feature_dimensions.append(featuremap_dims[i]['num_features'])
        
        self.ms_fuser1 = MultiScaleFusion(self.feature_dimensions[:-1])
        in_channels = [dim * 2 for dim in self.feature_dimensions[:-1]]
        self.ms_fuser2 = MultiScaleFusion(in_channels)
        self.attn_module = MultiSizeAttentionModule(in_channels, self.heights[:-1])
        
        self.up4_to_3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(self.feature_dimensions[3], self.feature_dimensions[2], kernel_size=3, padding=1),
                                 nn.BatchNorm2d(self.feature_dimensions[2]),
                                 nn.ReLU(inplace=True))
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(self.feature_dimensions[2] * 3, self.feature_dimensions[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dimensions[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dimensions[2], self.feature_dimensions[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dimensions[2]),
            nn.ReLU(inplace=True)
        )

        self.up3_to_2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(self.feature_dimensions[2], self.feature_dimensions[1], kernel_size=3, padding=1),
                                 nn.BatchNorm2d(self.feature_dimensions[1]),
                                 nn.ReLU(inplace=True))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(self.feature_dimensions[1] * 3, self.feature_dimensions[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dimensions[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dimensions[1], self.feature_dimensions[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dimensions[1]),
            nn.ReLU(inplace=True)
        )

        self.up2_to_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(self.feature_dimensions[1], self.feature_dimensions[0], kernel_size=3, padding=1),
                                 nn.BatchNorm2d(self.feature_dimensions[0]),
                                 nn.ReLU(inplace=True))
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.feature_dimensions[0] * 3, self.feature_dimensions[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dimensions[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dimensions[0], self.feature_dimensions[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dimensions[0]),
            nn.ReLU(inplace=True)
        )
        
        self.up1_to_0 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                 nn.Conv2d(self.feature_dimensions[0], self.feature_dimensions[0], kernel_size=3, padding=1),
                                 nn.BatchNorm2d(self.feature_dimensions[0]),
                                 nn.ReLU(inplace=True))

        self.conv_out = nn.Conv2d(self.feature_dimensions[0], num_classes, kernel_size=3, padding=1)
    
    def forward(self, images: Tensor, proto_features: List[Tensor]):
        with torch.no_grad():
            features = self.encoder(images)
        layer4_features = features[-1]
        features = features[:-1]
            
        pfeatures = get_prototype_features(features, proto_features)
        rfeatures = get_residual_features(features, pfeatures)
        
        # multi-scale fusion
        features = self.ms_fuser1(*features)
        rfeatures = self.ms_fuser1(*rfeatures)
        
        # concatenate the input features and the residual features
        cfeatures = get_concatenated_features(features, rfeatures)
        
        # attention modules
        features = self.attn_module(cfeatures)
        features = self.ms_fuser2(*features)
        
        # decoder
        layer3_features = self.up4_to_3(layer4_features)
        layer3_features = torch.cat([features[2], layer3_features], dim=1)
        layer3_features = self.conv_block3(layer3_features)
        
        layer2_features = self.up3_to_2(layer3_features)
        layer2_features = torch.cat([features[1], layer2_features], dim=1)
        layer2_features = self.conv_block2(layer2_features)
        
        layer1_features = self.up2_to_1(layer2_features)
        layer1_features = torch.cat([features[0], layer1_features], dim=1)
        layer1_features = self.conv_block1(layer1_features)

        out_features = self.up1_to_0(layer1_features)
        out = self.conv_out(out_features)
        
        return out


def dryrun_find_featuremap_dims(
    feature_extractor,
    input_size: tuple[int, int],
    num_layers: list[str],
    device: torch.device = torch.device('cuda')
) -> dict[str, int | tuple[int, int]]:
    
    dryrun_input = torch.empty(1, 3, *input_size, device=device)
    dryrun_features = feature_extractor(dryrun_input)
    
    featuremap_dims = []
    for i in range(num_layers):
        featuremap_dims.append({"num_features": dryrun_features[i].shape[1], "resolution": dryrun_features[i].shape[2:]})
    
    return featuremap_dims
    