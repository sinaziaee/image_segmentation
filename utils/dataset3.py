import os
import random
import numpy as np
import torch
import torch.utils.data as data
import SimpleITK as sitk


from PIL import Image

class SegmentationDataset(data.Dataset):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.gz']

    @staticmethod
    def _isimage(image, ends):
        return any(image.endswith(end) for end in ends)
    
    @staticmethod
    def _load_input_image(path):
        if str(path).endswith(".gz"):
            image = sitk.ReadImage(path)
            image_array = sitk.GetArrayFromImage(image)
            return image_array       
        else:  
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('L')
    
    @staticmethod
    def _load_target_image(path):
        if str(path).endswith(".gz"):
            image = sitk.ReadImage(path)
            image_array = sitk.GetArrayFromImage(image)
            return image_array
        else:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('L')
            
    def __init__(self, input_root, target_root, transform_input=None,
                 transform_target=None, seed_fn=None):
        assert bool(transform_input) == bool(transform_target)
        
        self.input_root = input_root
        self.target_root = target_root
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.seed_fn = seed_fn
                
        self.input_ids = sorted(img for img in os.listdir(self.input_root)
                                if self._isimage(img, self.IMG_EXTENSIONS))
        
        self.target_ids = sorted(img for img in os.listdir(self.target_root)
                                 if self._isimage(img, self.IMG_EXTENSIONS))
        
        assert(len(self.input_ids) == len(self.target_ids))
    
    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if self.seed_fn:
            self.seed_fn(seed)
        
    def __getitem__(self, idx):
        input_img = self._load_input_image(
            os.path.join(self.input_root, self.input_ids[idx]))
        target_img = self._load_target_image(
            os.path.join(self.target_root, self.target_ids[idx]))
        if self.transform_input:
            seed = random.randint(0, 2**32)
            self._set_seed(seed)
            input_img = self.transform_input(input_img)
            input_img = input_img.permute(1, 2, 0).contiguous()
            self._set_seed(seed)
            target_img = self.transform_target(target_img)
            target_img = target_img.permute(1, 2, 0).contiguous()
            
        return input_img, target_img
        
    def __len__(self):
        return len(self.input_ids)