import os
import torch
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.transforms import RandomHorizontalFlip


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BTAD(Dataset):
    
    CLASS_NAMES = ['01', '02', '03']
    
    def __init__(self, 
                 root: str,
                 class_name: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 **kwargs):
        #assert class_name in BTAD_CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, BTAD_CLASS_NAMES)
        self.root = root
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('crp_size'), kwargs.get('crp_size')]
        # load dataset
        if self.class_name is None:  # load all classes
            self.image_paths, self.labels, self.mask_paths, self.img_types = self._load_all_data()
            self.class_name = None
        else:
            self.image_paths, self.labels, self.mask_paths, self.img_types = self._load_data()
        # set transforms
        if train:
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size'), Image.ANTIALIAS),
                #T.RandomRotation(5),
                T.CenterCrop(kwargs.get('crp_size')),
                T.RandomHorizontalFlip(),
                T.ToTensor()])
        # test:
        else:
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size'), Image.ANTIALIAS),
                T.CenterCrop(kwargs.get('crp_size')),
                T.ToTensor()])
        # mask
        self.target_transform = T.Compose([
            T.Resize(kwargs.get('img_size'), Image.NEAREST),
            T.CenterCrop(kwargs.get('crp_size')),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize([0.5], [0.5])])
        
        self.class_to_idx = {'01': 0, '02': 1, '03': 2}
        self.idx_to_class = {0: '01', 1: '02', 2: '03'}

    def __getitem__(self, idx):
        image_path, label, mask, img_type = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.img_types[idx]

        image = Image.open(image_path).convert('RGB')
        
        image = self.normalize(self.transform(image))
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0],self.cropsize[0]])
        else:
            mask = Image.open(mask)
            mask = self.target_transform(mask)
        
        if self.train:
            if self.class_name is None:
                class_name = image_path.split('/')[-4]
            else:
                class_name = self.class_name
            label = self.class_to_idx[class_name]
        
        return image, label, mask, os.path.basename(image_path[:-4]), img_type
    
    def __len__(self):
        return len(self.image_paths)

    def _load_data(self):
        phase = 'train' if self.train else 'test'
        image_paths, labels, mask_paths, types = [], [], [], []

        img_dir = os.path.join(self.root, self.class_name, phase)
        gt_dir = os.path.join(self.root, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.bmp') or f.endswith('.png')])
            image_paths.extend(img_fpath_list)

            # load gt labels
            if img_type == 'ok':
                labels.extend([0] * len(img_fpath_list))
                mask_paths.extend([None] * len(img_fpath_list))
                types.extend(['ok'] * len(img_fpath_list))
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                if self.class_name == '03':
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.bmp')
                                 for img_fname in img_fname_list]
                else:
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                    for img_fname in img_fname_list]
                mask_paths.extend(gt_fpath_list)
                types.extend([img_type] * len(img_fpath_list))

        return image_paths, labels, mask_paths, types
    
    def _load_all_data(self):
        all_image_paths = []
        all_labels = []
        all_mask_paths = []
        all_types = []
        for class_name in self.CLASS_NAMES:
            self.class_name = class_name
            image_paths, labels, mask_paths, types = self._load_data()
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)
            all_mask_paths.extend(mask_paths)
            all_types.extend(types)
        return all_image_paths, all_labels, all_mask_paths, all_types
    
    def update_class_to_idx(self, class_to_idx):
        for class_name in self.class_to_idx.keys():
            self.class_to_idx[class_name] = class_to_idx[class_name]
        class_names = self.class_to_idx.keys()
        idxs = self.class_to_idx.values()
        self.idx_to_class = dict(zip(idxs, class_names))