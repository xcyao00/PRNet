import os
from typing import List
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


def get_prototype_features(features: List[Tensor], proto_features: List[Tensor]) -> List[Tensor]:
    matched_proto_features = []
    for layer_id in range(len(features)):
        fi = features[layer_id]  # (B, dim, h, w)
        pi = proto_features[layer_id]  # (K, dim, h, w)
        B, C, H, W = fi.shape
        K, _, _, _ = pi.shape
        fir = fi.unsqueeze(1).expand(B, K, C, H, W).reshape(-1, C, H, W)
        pir = pi.unsqueeze(0).expand(B, K, C, H, W).reshape(-1, C, H, W)
        fir = fir.reshape(B * K, -1)
        pir = pir.reshape(B * K, -1)
        l2_dist = F.pairwise_distance(fir, pir, p=2)
        seps = l2_dist.chunk(B)
        cats = torch.stack(seps, dim=0)  # (B, K)
        inds = torch.argmin(cats, dim=1)  # (B, )
        matched_pi = pi[inds]  # (B, dim, h, w)
        matched_proto_features.append(matched_pi)
    
    return matched_proto_features


def get_residual_features(features: List[Tensor], proto_features: List[Tensor]) -> List[Tensor]:
    residual_features = []
    for layer_id in range(len(features)):
        fi = features[layer_id]  # (B, dim, h, w)
        pi = proto_features[layer_id]  # (B, dim, h, w)
        
        ri = F.mse_loss(fi, pi, reduction='none')
        residual_features.append(ri)
    
    return residual_features


def get_concatenated_features(features1: List[Tensor], features2: List[Tensor]) -> List[Tensor]:
    cfeatures = []
    for layer_id in range(len(features1)):
        fi = features1[layer_id]  # (B, dim, h, w)
        pi = features2[layer_id]  # (B, dim, h, w)
        
        ci = torch.cat([fi, pi], dim=1)
        cfeatures.append(ci)
    
    return cfeatures
        

def load_prototype_features(root_dir: str, class_name: str, device: torch.device) -> List[Tensor]:
    layer1_protos = np.load(os.path.join(root_dir, class_name, 'layer1.npy'))
    layer2_protos = np.load(os.path.join(root_dir, class_name, 'layer2.npy'))
    layer3_protos = np.load(os.path.join(root_dir, class_name, 'layer3.npy'))
    
    layer1_protos = torch.from_numpy(layer1_protos).to(device)
    layer2_protos = torch.from_numpy(layer2_protos).to(device)
    layer3_protos = torch.from_numpy(layer3_protos).to(device)
    
    return layer1_protos, layer2_protos, layer3_protos