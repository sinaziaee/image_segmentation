import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import random
import torch
from utils import unet
import cv2 as cv

class CustomDataset(Dataset):
    def __init__(self, root_dir, base_image_paths='Training_Images',
                base_label_paths='Ground_Truth', transform=None, spatial_transforms=None, non_spatial_transforms=None):
        self.root_dir = root_dir
        self.transform = transform
        self.spatial_transforms = spatial_transforms
        self.non_spatial_transforms = non_spatial_transforms
        self.base_image_paths = os.path.join(self.root_dir, base_image_paths)
        self.base_label_paths = os.path.join(self.root_dir, base_label_paths)

        self.image_paths = []
        self.label_paths = []
        self.images_no = 0
        self.masks_no = 0

        for img in sorted(os.listdir(self.base_image_paths)):
            if 'png' in str(img):
                image_path = os.path.join(self.base_image_paths, img)
                self.image_paths.append(image_path)
                #label_path2 ="training_label_"+image_path.split('.')[0].split('_')[2]+image_path.split('.')[0].split('_')[3]
                
        for lbl in sorted(os.listdir(self.base_label_paths)):
            if 'png' in str(lbl):
                label_path = os.path.join(self.base_label_paths, lbl)
                self.label_paths.append(label_path)

        total_samples = len(self.image_paths)
        total_labels = len(self.label_paths)

        assert total_samples == total_labels, f"Number of images and labels don't match. imgs:{total_samples}, lbls:{total_labels}"
        self.images_no = total_samples
        self.labels_no = total_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]

        image = self.load_image(image_path)
        label = self.load_image(label_path)

        if self.spatial_transforms is not None:
            augmented = self.spatial_transforms(image=image, label=label)
            image = augmented['image']
            label = augmented['label']
            if self.non_spatial_transforms is not None:
                augmented_image = self.non_spatial_transforms(image=image)
                image = augmented_image['image']
        else:
            image = self.transform(image)
            label = self.transform(label)
        
        # print(type(image))

        return image, label

    def load_image(self, path):
        # image = Image.open(path).convert("L")
        image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        image = np.array(image)
        # image = np.expand_dims(image, axis=0)

        return image

def create_data_loaders(path_dir, image_dir, label_dir, data_transformer, batch_size=16, 
                        split_size=[0.8, 0.1], spatial_transforms=None, non_spatial_transforms=None):
    dataset = CustomDataset(root_dir=path_dir, base_image_paths=image_dir,
                                    base_label_paths=label_dir, transform=data_transformer, 
                                    spatial_transforms=spatial_transforms, non_spatial_transforms=non_spatial_transforms)
    
    if split_size is not None:
        train_size = int(split_size[0] * len(dataset))
        valid_size = int(split_size[1] * len(dataset))
        test_size = len(dataset) - train_size - valid_size
        # batch_size = 32
        train_dataset, valid_dataset, test_dataset = random_split(dataset,
                                                [train_size, valid_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f'dataset info: \n No images: {dataset.images_no}, No masks: {dataset.labels_no}, \n Loaders Len: t:{len(train_loader)}, v:{len(valid_loader)}, test: {len(test_loader)}')
        return train_loader, valid_loader, test_loader
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f'dataset info: \n No images: {dataset.images_no}, No masks: {dataset.labels_no}, \n No of batches: {len(loader)}, batch shape: {next(iter(loader))[0].shape}')
        # print(f'dataset info: \n No images: {dataset.images_no}, No masks: {dataset.labels_no}, \n No of batches: {len(loader)}')
        return loader
    