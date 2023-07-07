import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.mvtec import TRAINMVTEC, MVTEC
from models.prnet import PRNet
from losses.focal_loss import FocalLoss
from losses.smooth_l1_loss import SmoothL1Loss
from utils import load_prototype_features
from test import validate


def main(args, class_name):
    train_dataset = TRAINMVTEC(args.data_path, args.anomaly_source_path, class_name=class_name, train=True, img_size=256, crp_size=256,
                          msk_size=256, msk_crp_size=256, num_anomalies=args.num_anomalies)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    test_dataset = MVTEC(args.data_path, class_name=class_name, train=False, img_size=256, crp_size=256,
                          msk_size=256, msk_crp_size=256)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False
    )
    
    model = PRNet(args.backbone, num_classes=1, device=args.device).to(args.device)
    smooth_l1_loss = SmoothL1Loss()
    focal_loss = FocalLoss(alpha=0.5, gamma=4)
    
    proto_features = load_prototype_features(args.proto_path, class_name, args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_img_auc, img_epoch = 0, 0
    best_pix_auc, pix_epoch = 0, 0
    for epoch in range(args.epochs):
        model.train()
        train_loss_total, total_num = 0, 0
        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}]")
        for step, batch in enumerate(train_loader):
            progress_bar.update(1)
            images, _, mask = batch
            
            images = images.to(args.device)
            mask = mask.to(args.device)
            
            logits = model(images, proto_features)
            scores = torch.sigmoid(logits)
            
            loss1 = smooth_l1_loss(scores, mask)
            scores = torch.cat([1 - scores, scores], dim=1)
            loss2 = focal_loss(scores, mask)
            loss = loss1 + args.weight_lambda * loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
            total_num += 1
               
        progress_bar.close()
        print(f"EpochEpoch[{epoch}/{args.epochs}]: train_loss: {train_loss_total / total_num}")
        
        if (args.eval_freq > 0) and ((epoch + 1) % args.eval_freq == 0):
            
            img_auc, pix_auc = validate(model, test_loader, proto_features, args.device)
            print("Epoch: {}, Class Name: {}, Image AUC: {} Pixel AUC: {}".format(epoch, class_name, img_auc, pix_auc))
            
            os.makedirs(os.path.join(args.checkpoint_path, class_name), exist_ok=True)  
            if img_auc > best_img_auc:
                best_img_auc = img_auc
                ckpt_path = os.path.join(args.checkpoint_path, class_name, args.backbone + "-image-level.pth")
                torch.save(model.state_dict(), ckpt_path)
                img_epoch = epoch
            if pix_auc > best_pix_auc:
                best_pix_auc = pix_auc
                ckpt_path = os.path.join(args.checkpoint_path, class_name, args.backbone + "-pixel-level.pth")
                torch.save(model.state_dict(), ckpt_path)
                pix_epoch = epoch
                
    return best_img_auc, best_pix_auc, img_epoch, pix_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/data2/yxc/datasets/mvtec_anomaly_detection/")
    parser.add_argument('--anomaly_source_path', type=str, default="/data2/yxc/datasets/dtd/images/")
    parser.add_argument('--proto_path', type=str, default="./prototypes")
    parser.add_argument('--class_name', type=str, default='all')
    parser.add_argument('--num_anomalies', type=int, default=10)
    
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--gpu_id', type=str, default="1")
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--backbone', type=str, default="resnet18")
    parser.add_argument('--weight_lambda', type=float, default=5)
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    if args.class_name == 'all':
        image_aucs, pixel_aucs = [], []
        for class_name in MVTEC.CLASS_NAMES:
            img_auc, pix_auc, img_epoch, pix_epoch = main(args, class_name)
            image_aucs.append(img_auc)
            pixel_aucs.append(pix_auc)
        for i, class_name in enumerate(MVTEC.CLASS_NAMES):
            print("{}: Best, Image AUC: {} Pixel AUC: {}".format(class_name, image_aucs[i], pixel_aucs[i]))
        print("Best Mean, Image AUC: {} Pixel AUC: {}".format(np.mean(image_aucs), np.mean(pixel_aucs)))
    else:
        img_auc, pix_auc, img_epoch, pix_epoch = main(args, args.class_name)
        print("{}: Best, Image AUC: {} Pixel AUC: {}".format(args.class_name, img_auc, pix_auc))
    

    
    
            