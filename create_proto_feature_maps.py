import os
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import timm
import torch
from torch.utils.data import DataLoader
from datasets.mvtec import MVTEC


device = 'cuda'
root_dir = '/data2/yxc/datasets/mvtec_anomaly_detection'


def create_proto_feature_maps(class_name, ratio=0.1):
    train_dataset = MVTEC(root_dir, class_name=class_name, train=True, img_size=256, crp_size=256,
                          msk_size=256, msk_crp_size=256)
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False
    )
    encoder = timm.create_model("resnet18", features_only=True, 
            out_indices=(1, 2, 3), pretrained=True).eval()
    encoder = encoder.to(device)
    
    progress_bar = tqdm(total=len(train_loader))
    progress_bar.set_description(f"Extract Features")
    layer1_features, layer2_features, layer3_features = [], [], []
    for step, batch in enumerate(train_loader):
        progress_bar.update(1)
        images, _, _, _, _ = batch
        
        images = images.to(device)
        
        with torch.no_grad():
            features = encoder(images)
        layer1_features.append(features[0].cpu().numpy())
        layer2_features.append(features[1].cpu().numpy())
        layer3_features.append(features[2].cpu().numpy())    
    progress_bar.close()
    
    N = len(train_dataset)
    layer1_features = np.concatenate(layer1_features, axis=0)
    layer2_features = np.concatenate(layer2_features, axis=0)
    layer3_features = np.concatenate(layer3_features, axis=0)
    _, C1, H1, W1 = layer1_features.shape
    _, C2, H2, W2 = layer2_features.shape
    _, C3, H3, W3 = layer3_features.shape
    layer1_features = layer1_features.reshape(N, -1)
    layer2_features = layer2_features.reshape(N, -1)
    layer3_features = layer3_features.reshape(N, -1)
    
    K = int(len(train_dataset) * ratio)
    
    print("fitting layer1...")
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(layer1_features)
    layer1_features = kmeans.cluster_centers_
    layer1_features = layer1_features.reshape(K, C1, H1, W1)
    
    print("fitting layer2...")
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(layer2_features)
    layer2_features = kmeans.cluster_centers_
    layer2_features = layer2_features.reshape(K, C2, H2, W2)
    
    print("fitting layer3...")
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(layer3_features)
    layer3_features = kmeans.cluster_centers_
    layer3_features = layer3_features.reshape(K, C3, H3, W3)
    
    os.makedirs(os.path.join('prototypes', class_name), exist_ok=True)
    np.save(os.path.join('prototypes', class_name, 'layer1.npy'), layer1_features)
    np.save(os.path.join('prototypes', class_name, 'layer2.npy'), layer2_features)
    np.save(os.path.join('prototypes', class_name, 'layer3.npy'), layer3_features)


if __name__ == '__main__':
    for class_name in MVTEC.CLASS_NAMES:
        create_proto_feature_maps(class_name)
    

    
    
            