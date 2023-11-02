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
                base_label_paths='Ground_Truth', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.base_image_paths = os.path.join(self.root_dir, base_image_paths)
        self.base_label_paths = os.path.join(self.root_dir, base_label_paths)

        self.image_paths = []
        self.label_paths = []

        for img in sorted(os.listdir(self.base_image_paths)):
            if 'jpg' in str(img):
                image_path = os.path.join(self.base_image_paths, img)
                self.image_paths.append(image_path)

        for lbl in sorted(os.listdir(self.base_label_paths)):
            if 'png' in str(lbl):
                label_path = os.path.join(self.base_label_paths, lbl)
                self.label_paths.append(label_path)

        total_samples = len(self.image_paths)
        total_labels = len(self.label_paths)

        assert total_samples == total_labels, "Number of images and labels don't match."

        assert len(self.image_paths) == len(self.label_paths), "Number of filtered images and labels don't match."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]

        image = self.load_image(image_path)
        label = self.load_image(label_path)

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def load_image(self, path):
        # image = Image.open(path).convert("L")
        image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        # image = np.expand_dims(image, axis=0)

        return image
    
def create_transformer(img_size=320):
    data_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    return data_transformer

def create_data_loaders(path_dir, image_dir, label_dir, data_transformer):
    dataset = CustomDataset(root_dir=path_dir, base_image_paths=image_dir,
                                    base_label_paths=label_dir, transform=data_transformer)

    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    batch_size = 32
    train_dataset, valid_dataset, test_dataset = random_split(dataset,
                                            [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader