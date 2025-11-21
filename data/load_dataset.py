import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image 
from config import * 
import torchvision.transforms as T


class SegDataset(Dataset):
    def __init__(self, dataset, images_dir, transform_sample):
        image_root_path = images_dir + '/pic'
        mask_root_path = images_dir + '/mask'
        
        self.image_slices = []
        self.mask_slices = []
        
        self.transform_sample = transform_sample
        if dataset == 'InES':
            transform_target = T.Compose([
                T.ToTensor(),
                T.Resize([256,256]),
            ])
        if dataset == 'CVC':
            transform_target = T.Compose([
                T.ToTensor(),
                T.Resize([256,256]),
                T.Grayscale(1),
            ])
        
        for im_name in os.listdir(image_root_path):
            mask_name = im_name.split('.')[0] + '.png'
            image_path = image_root_path + "/" + im_name
            mask_path = mask_root_path + "/" + mask_name

            im = Image.open(image_path)

            mask = np.asarray(transform_target(Image.open(mask_path)))
            background = (mask == 0).astype(np.float32)
            output_tensor = np.zeros((2, 256, 256), dtype=np.float32)
            output_tensor[0, :, :] = background
            output_tensor[1, :, :] = 1 - background

            self.image_slices.append(im) # image set
            self.mask_slices.append(output_tensor) # mask set

        self.moving_prob = np.zeros((len(self.image_slices), 2, 256, 256), dtype=np.float32) # for TiDAL

    def __getitem__(self, idx):
        image = self.image_slices[idx] 
        mask = self.mask_slices[idx]

        image1 = self.transform_sample['ori'](image) # original img
        image2 = self.transform_sample['aug'](image) # augmented img

        moving_prob = self.moving_prob[idx] # for TiDAL

        return image1, image2, mask, idx, moving_prob

    def __len__(self):
        return len(self.image_slices)
    

# Data
def load_dataset(args):
    dataset = args.dataset

    transforms = {
        'ori' : T.Compose([
            T.ToTensor(), 
            T.Resize([256,256]), 
            ]),
        'aug' : T.Compose([
            T.ToTensor(),
            T.Resize([256,256]),
            T.ColorJitter(brightness=BRI, contrast=CON, saturation=SAT, hue=HUE), # color aug
            # T.GaussianBlur(kernel_size=5, sigma=(0.0, 1.0)), # noise aug
            ]),
    }

    if dataset == 'InES':
        data_train = SegDataset('InES', 'data/InES/train', transforms)
        data_unlabeled = SegDataset('InES', 'data/InES/train', transforms)
        data_valid = SegDataset('InES', 'data/InES/valid', transforms)
        data_test = SegDataset('InES', 'data/InES/test', transforms)
    elif dataset == 'CVC':
        data_train = SegDataset('CVC', 'data/CVC/train', transforms)
        data_unlabeled = SegDataset('CVC', 'data/CVC/train', transforms)
        data_valid = SegDataset('CVC', 'data/CVC/valid', transforms)
        data_test = SegDataset('CVC', 'data/CVC/test', transforms)
    elif dataset == 'InES2CVC':
        data_train = SegDataset('InES', 'data/InES/train', transforms)
        data_unlabeled = SegDataset('InES', 'data/InES/train', transforms)
        data_valid = SegDataset('CVC', 'data/CVC/valid', transforms)
        data_test = SegDataset('CVC', 'data/CVC/test', transforms)
        
    return data_train, data_unlabeled, data_valid, data_test, args.add_num, len(data_train)