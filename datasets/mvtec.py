import os
import random
from typing import Any, Callable, Optional, Tuple
import torch
import tarfile
import numpy as np
import urllib.request
from PIL import Image
from torchvision.transforms.transforms import RandomHorizontalFlip
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as T

import cv2
import glob
import imgaug.augmenters as iaa
import albumentations as A
from .perlin import rand_perlin_2d_np
from skimage.segmentation import mark_boundaries


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MVTEC(Dataset):
    """`MVTEC <>`_ Dataset.

    Args:
        root (string): Root directory of dataset, i.e ``../../mvtec_anomaly_detection``.
        train (bool, optional): If True, creates dataset for training, otherwise for testing.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.Resize``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    MVTEC_URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
    
    CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    def __init__(
            self, 
            root: str,
            class_name: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            **kwargs):
        
        self.root = root
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('msk_crp_size'), kwargs.get('msk_crp_size')]
        self.pseudo_anomaly = False
        
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
                T.CenterCrop(kwargs.get('crp_size')),
                #T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        else:
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size'), Image.ANTIALIAS),
                T.CenterCrop(kwargs.get('crp_size')),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        # mask
        self.target_transform = T.Compose([
            T.Resize(kwargs.get('msk_size'), Image.NEAREST),
            T.CenterCrop(kwargs.get('msk_crp_size')),
            T.ToTensor()])
            
        self.class_to_idx = {'bottle': 0, 'cable': 1, 'capsule': 2, 'carpet': 3,
                'grid': 4, 'hazelnut': 5, 'leather': 6, 'metal_nut': 7,
                'pill': 8, 'screw': 9, 'tile': 10, 'toothbrush': 11,
                'transistor': 12, 'wood': 13, 'zipper': 14}
        self.idx_to_class = {0: 'bottle', 1: 'cable', 2: 'capsule', 3: 'carpet',
                        4: 'grid', 5: 'hazelnut', 6: 'leather', 7: 'metal_nut',
                        8: 'pill', 9: 'screw', 10: 'tile', 11: 'toothbrush',
                        12: 'transistor', 13: 'wood', 14: 'zipper'}

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, label, mask) where label is 0 for normal image and 1 for abnormal image t.
        """
        image_path, label, mask, img_type = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.img_types[idx]
        
        if self.class_name is None:
            class_name = image_path.split('/')[-4]
        else:
            class_name = self.class_name
            
        image = Image.open(image_path)
        if class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            image = np.expand_dims(np.array(image), axis=2)
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        image = self.transform(image)
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.target_transform(mask)
        
        if self.train:
            label = self.class_to_idx[class_name]
        
        return image, label, mask, os.path.basename(image_path[:-4]), img_type

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self):
        phase = 'train' if self.train else 'test'
        image_paths, labels, mask_paths, types = [], [], [], []

        image_dir = os.path.join(self.root, self.class_name, phase)
        mask_dir = os.path.join(self.root, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(image_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(image_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            image_paths.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                labels.extend([0] * len(img_fpath_list))
                mask_paths.extend([None] * len(img_fpath_list))
                types.extend(['good'] * len(img_fpath_list))
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(mask_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
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


class TRAINMVTEC(Dataset):
    """
    Mvtec train dataset with anomaly samples.
    Anomaly samples: real anomaly samples,
        anomaly samples generated by copy-pasting abnormal regions to normal samples
    """
    def __init__(self, 
                 data_path: str,
                 anomaly_source_path: str,
                 class_name: str = 'bottle', 
                 train: bool = True,
                 in_fg_region: bool = True,
                 **kwargs) -> None:
    
        self.dataset_path = data_path
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('crp_size', 256), kwargs.get('crp_size', 256)]
        # number of total used real anomaly samples
        self.num_load_anomalies = kwargs.get('num_anomalies', 10)  
        self.repeat_num = 10  # repeat times for anomaly samples
        self.reuse_times = 5  # real anomaly reuse times
        self.in_fg_region = in_fg_region
        
        # load dataset
        self.n_imgs, self.n_labels, self.n_masks, self.a_imgs, self.a_labels, self.a_masks = self.load_dataset_folder()
        
        # number of pseudo and real anomalies togather matches the normal images
        # so in each bacth, the number of normal images and anomaly images will be about equal.
        # this can make normal and anomaly images in each batch to be balanced.
        self.num_pseudo_anomalies = self.num_real_anomalies = len(self.n_imgs) // 2
        
        # repeat the real anomalies
        a_imgs, a_labels, a_masks = [], [], []
        N = len(self.a_imgs)
        for i in range(self.num_real_anomalies):
            a_imgs.append(self.a_imgs[i % N])
            a_labels.append(self.a_labels[i % N])
            a_masks.append(self.a_masks[i % N])
        self.a_imgs = a_imgs
        self.a_labels = a_labels
        self.a_masks = a_masks
        
        # set transforms
        if train:
            self.transform_img = T.Compose([
                T.Resize(kwargs.get('img_size', 256), Image.ANTIALIAS),
                #T.RandomRotation(5),
                T.CenterCrop(kwargs.get('crp_size', 256)),
                T.ToTensor()])
        # test:
        else:
            self.transform_img = T.Compose([
                T.Resize(kwargs.get('img_size', 256), Image.ANTIALIAS),
                T.CenterCrop(kwargs.get('crp_size', 256)),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(kwargs.get('msk_size', 256), Image.NEAREST),
            T.CenterCrop(kwargs.get('msk_crp_size', 256)),
            T.ToTensor()])
        
        self.augmentors_for_real = [A.RandomRotate90(),
                A.Flip(),
                A.Transpose(),
                A.OpticalDistortion(p=1.0, distort_limit=1.0),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(),
                    A.GaussNoise(),
                ], p=0.2),
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=.1),
                    A.IAAPiecewiseAffine(p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.IAASharpen(),
                    A.IAAEmboss(),
                    A.RandomBrightnessContrast(),            
                ], p=0.3),
                A.HueSaturationValue(p=0.3)]

        self.normalize = T.Compose([T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        
        # For pseudo anomalies generation
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))
        self.augmentors_for_pseudo = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        # keep same with the MVTecCopyPasteDataset
        self.transform_img_np = T.Compose([
                T.Resize(kwargs.get('img_size', 256), Image.ANTIALIAS),
                #T.RandomRotation(5),
                T.CenterCrop(kwargs.get('crp_size', 256))])
        self.normalize_np = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    
    def __len__(self):
        return len(self.n_imgs) + len(self.a_imgs) + self.num_pseudo_anomalies

    def __getitem__(self, idx):
        if idx >= len(self.n_imgs):  # anomaly samples
            # pseudo anomalies
            if idx >= len(self.n_imgs) + len(self.a_imgs):  
                n_idx = np.random.randint(len(self.n_imgs))  # get a random normal sample
                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
                img, label, mask = self.transform_image(self.n_imgs[n_idx], 
                                            self.anomaly_source_paths[anomaly_source_idx])
            # real and augmented real anomalies
            else:
                idx_ = idx - len(self.n_imgs)
                # real anomaly paths
                img, label, mask = self.a_imgs[idx_], self.a_labels[idx_], self.a_masks[idx_]
                # augmented real anomalies
                if idx >= len(self.n_imgs) + self.num_load_anomalies * self.reuse_times:
                    # generating anomaly sample by copy-pasting
                    img, mask = self.copy_paste(img, mask)
                    img, mask = Image.fromarray(img), Image.fromarray(mask)
                    img = self.normalize(self.transform_img(img))
                    mask = self.transform_mask(mask)
                # directly load real anomalies
                else:
                    img, label, mask = self._load_image_and_mask(img, label, mask)

            return img, label, mask
        else:  # normal samples
            img, label, mask = self.n_imgs[idx], self.n_labels[idx], self.n_masks[idx]
            img, label, mask = self._load_image_and_mask(img, label, mask)
            
            return img, label, mask
    
    def _load_image_and_mask(self, img_path, label, mask_path):
        img = Image.open(img_path)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            img = np.expand_dims(np.asarray(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        #
        img = self.normalize(self.transform_img(img))
        #
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask_path)
            mask = self.transform_mask(mask)
        
        return img, label, mask
    
    def realRandAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmentors_for_real)), 3, replace=False)
        aug = A.Compose([self.augmentors_for_real[aug_ind[0]],
                         self.augmentors_for_real[aug_ind[1]],
                         self.augmentors_for_real[aug_ind[2]]])
        return aug
    
    def pseudoRandAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmentors_for_pseudo)), 3, replace=False)
        aug = iaa.Sequential([self.augmentors_for_pseudo[aug_ind[0]],
                              self.augmentors_for_pseudo[aug_ind[1]],
                              self.augmentors_for_pseudo[aug_ind[2]]])
        return aug

    def copy_paste(self, img, mask):
        n_idx = np.random.randint(len(self.n_imgs))  # get a random normal sample
        aug = self.realRandAugmenter()

        image = cv2.imread(img)  # anomaly sample
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (900, 900, 3)
        n_image = cv2.imread(self.n_imgs[n_idx])  # normal sample
        n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB)  # (900, 900, 3)
        img_height, img_width = n_image.shape[0], n_image.shape[1]

        mask = Image.open(mask)
        mask = np.asarray(mask)  # (900, 900)
        
        # augmente the abnormal region
        augmentated = aug(image=image, mask=mask)
        aug_image, aug_mask = augmentated['image'], augmentated['mask']
        if self.in_fg_region:
            n_img_path = self.n_imgs[n_idx]
            img_file = n_img_path.split('/')[-1]
            fg_path = os.path.join(f'fg_mask/{self.class_name}', img_file)
            fg_mask = Image.open(fg_path)
            fg_mask = np.asarray(fg_mask)
            
            intersect_mask = np.logical_and(fg_mask == 255, aug_mask == 255)
            if (np.sum(intersect_mask) > int(2 / 3 * np.sum(aug_mask == 255))):
                # when most part of aug_mask is in the fg_mask region 
                # copy the augmentated anomaly area to the normal image
                n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                return n_image, aug_mask
            else:
                contours, _ = cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                center_xs, center_ys = [], []
                widths, heights = [], []
                for i in range(len(contours)):
                    M = cv2.moments(contours[i])
                    if M['m00'] == 0:  # error case
                        x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                        y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                        center_x = int((x_min + x_max) / 2)
                        center_y = int((y_min + y_max) / 2)
                    else:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                    center_xs.append(center_x)
                    center_ys.append(center_y)
                    x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                    y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                    width, height = x_max - x_min, y_max - y_min
                    widths.append(width)
                    heights.append(height)
                if len(widths) == 0 or len(heights) == 0:  # no contours
                    n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                    return n_image, aug_mask
                else:
                    max_width, max_height = np.max(widths), np.max(heights)
                    center_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    center_mask[int(max_height/2):img_height-int(max_height/2), int(max_width/2):img_width-int(max_width/2)] = 255
                    fg_mask = np.logical_and(fg_mask == 255, center_mask == 255)

                    x_coord = np.arange(0, img_width)
                    y_coord = np.arange(0, img_height)
                    xx, yy = np.meshgrid(x_coord, y_coord)
                    # coordinates of fg region points
                    xx_fg = xx[fg_mask]
                    yy_fg = yy[fg_mask]
                    xx_yy_fg = np.stack([xx_fg, yy_fg], axis=-1)  # (N, 2)
                    
                    if xx_yy_fg.shape[0] == 0:  # no fg
                        n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                        return n_image, aug_mask

                    aug_mask_shifted = np.zeros((img_height, img_width), dtype=np.uint8)
                    for i in range(len(contours)):
                        aug_mask_shifted_i = np.zeros((img_height, img_width), dtype=np.uint8)
                        new_aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                        # random generate a point in the fg region
                        idx = np.random.choice(np.arange(xx_yy_fg.shape[0]), 1)
                        rand_xy = xx_yy_fg[idx]
                        delta_x, delta_y = center_xs[i] - rand_xy[0, 0], center_ys[i] - rand_xy[0, 1]
                        
                        x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                        y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                        
                        # mask for one anomaly region
                        aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                        aug_mask_i[y_min:y_max, x_min:x_max] = 255
                        aug_mask_i = np.logical_and(aug_mask == 255, aug_mask_i == 255)
                        
                        # coordinates of orginal mask points
                        xx_ano, yy_ano = xx[aug_mask_i], yy[aug_mask_i]
                        
                        # shift the original mask into fg region
                        xx_ano_shifted = xx_ano - delta_x
                        yy_ano_shifted = yy_ano - delta_y
                        outer_points_x = np.logical_or(xx_ano_shifted < 0, xx_ano_shifted >= img_width) 
                        outer_points_y = np.logical_or(yy_ano_shifted < 0, yy_ano_shifted >= img_height)
                        outer_points = np.logical_or(outer_points_x, outer_points_y)
                        
                        # keep points in image
                        xx_ano_shifted = xx_ano_shifted[~outer_points]
                        yy_ano_shifted = yy_ano_shifted[~outer_points]
                        aug_mask_shifted_i[yy_ano_shifted, xx_ano_shifted] = 255
                        
                        # original points should be changed
                        xx_ano = xx_ano[~outer_points]
                        yy_ano = yy_ano[~outer_points]
                        new_aug_mask_i[yy_ano, xx_ano] = 255
                        # copy the augmentated anomaly area to the normal image
                        n_image[aug_mask_shifted_i == 255, :] = aug_image[new_aug_mask_i == 255, :]
                        aug_mask_shifted[aug_mask_shifted_i == 255] = 255
                    return n_image, aug_mask_shifted
        else:  # no fg restriction
            # copy the augmentated anomaly area to the normal image
            n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]

            return n_image, aug_mask

    def augment_image(self, image, anomaly_source_path):
        aug = self.pseudoRandAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.cropsize[1], self.cropsize[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.cropsize[0], self.cropsize[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)
        
        augmented_image = augmented_image.astype(np.float32)
        mask = (perlin_thr).astype(np.float32)
        augmented_image = mask * augmented_image + (1 - mask) * image
        label = 1
        if np.sum(mask) == 0:
            label = 0
        return augmented_image, label, mask

    def transform_image(self, image_path, anomaly_source_path):
        image = Image.open(image_path)
        
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            image = np.expand_dims(np.asarray(image), axis=2)
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        #
        image = self.transform_img_np(image)
        image = np.asarray(image)  # (256, 256, 3)

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        augmented_image, label, mask = self.augment_image(image, anomaly_source_path)
        augmented_image = augmented_image.astype(np.uint8)
        augmented_image = Image.fromarray(augmented_image)
        augmented_image = self.normalize_np(augmented_image)
        mask = torch.from_numpy(np.transpose(mask, (2, 0, 1)))

        return augmented_image, label, mask

    def load_dataset_folder(self):
        n_img_paths, n_labels, n_mask_paths = [], [], []  # normal
        a_img_paths, a_labels, a_mask_paths = [], [], []  # abnormal

        img_dir = os.path.join(self.dataset_path, self.class_name, 'test')
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        ano_types = sorted(os.listdir(img_dir))  # anomaly types

        num_ano_types = len(ano_types) - 1
        anomaly_nums_per_type = self.num_load_anomalies // num_ano_types
        extra_nums = self.num_load_anomalies % num_ano_types
        extra_ano_img_list, extra_ano_gt_list = [], []
        for type_ in ano_types:
            # load images
            img_type_dir = os.path.join(img_dir, type_)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])

            if type_ == 'good':  # normal images
                continue
            else:  # anomaly images
                # randomly choose some anomaly images
                random.shuffle(img_fpath_list)
                a_img_paths.extend(img_fpath_list[:anomaly_nums_per_type])
                a_labels.extend([1] * anomaly_nums_per_type)

                extra_ano_img_list.extend(img_fpath_list[anomaly_nums_per_type:])

                gt_type_dir = os.path.join(gt_dir, type_)
                ano_img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list[:anomaly_nums_per_type]]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in ano_img_fname_list]
                a_mask_paths.extend(gt_fpath_list)

                extra_img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list[anomaly_nums_per_type:]]
                extra_gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in extra_img_fname_list]
                extra_ano_gt_list.extend(extra_gt_fpath_list)
        if extra_nums > 0:
            assert len(extra_ano_img_list) == len(extra_ano_gt_list)
            inds = list(range(len(extra_ano_img_list)))
            random.shuffle(inds)
            select_ind = inds[:extra_nums]
            extra_a_img_paths = [extra_ano_img_list[ind] for ind in select_ind]
            extra_a_labels = [1] * extra_nums
            extra_a_mask_paths = [extra_ano_gt_list[ind] for ind in select_ind]
            a_img_paths.extend(extra_a_img_paths)
            a_labels.extend(extra_a_labels)
            a_mask_paths.extend(extra_a_mask_paths)

        # append normal images in train set
        img_dir = os.path.join(self.dataset_path, self.class_name, 'train', 'good')
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                    for f in os.listdir(img_dir)
                                    if f.endswith('.png')])
        n_img_paths.extend(img_fpath_list)
        n_labels.extend([0] * len(img_fpath_list))
        n_mask_paths.extend([None] * len(img_fpath_list))
        
        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths