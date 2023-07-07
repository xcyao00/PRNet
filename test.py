from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def validate(model, test_loader, proto_features, device):
    model.eval()
    
    label_list, gt_mask_list, score_list = [], [], []
    progress_bar = tqdm(total=len(test_loader))
    progress_bar.set_description(f"Evaluating")
    for step, batch in enumerate(test_loader):
        progress_bar.update(1)
        image, label, mask, _, _ = batch
            
        gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
        label_list.append(label.cpu().numpy().astype(bool).ravel())
        
        image = image.to(device)
        
        with torch.no_grad():
            logits = model(image, proto_features)
        scores = torch.sigmoid(logits)
        score_list.append(scores.squeeze(1).cpu())    
    progress_bar.close()
    
    labels = np.concatenate(label_list)
    gt_mask = np.concatenate(gt_mask_list, axis=0)
    scores = torch.cat(score_list, dim=0)
    
    img_scores = torch.topk(scores.reshape(scores.shape[0], -1), 100, dim=1)[0]
    img_scores = torch.mean(img_scores, dim=1)
    img_scores = img_scores.cpu().numpy()
    scores = scores.cpu().numpy()
            
    image_auc = round(roc_auc_score(labels, img_scores), 3)
    pixel_auc = round(roc_auc_score(gt_mask.flatten(), scores.flatten()), 3)

    return image_auc, pixel_auc