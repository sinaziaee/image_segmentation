import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from torchvision import transforms
import nibabel as nib
import SimpleITK as sitk


def show_image(image,mask,pred_image = None, path_dir=None, num=None):
    
    if pred_image == None:
        
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        
        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1,2,0).squeeze(),cmap = 'gray')
        
        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1,2,0).squeeze(),cmap = 'gray')
        
    elif pred_image != None :
        
        f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,5))
        
        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1,2,0).squeeze(),cmap = 'gray')
        
        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1,2,0).squeeze(),cmap = 'gray')
        
        ax3.set_title('MODEL OUTPUT')
        ax3.imshow(pred_image.permute(1,2,0).squeeze(),cmap = 'gray')
        # Save the figure if a save_file path is provided
        if path_dir is not None:
            plt.savefig(f'{path_dir}/{num}.png')


def visualize_training(train_loss_list, valid_loss_list, valid_iou_list=None, dice_score_list=None, valid_dice_list=None, results_folder= None):
    ax, fig = plt.subplots(1, 3, figsize=(16, 5))
    fig[0].plot(train_loss_list, color='blue', label='Train Loss')
    fig[0].plot(valid_loss_list, color='orange', label='Valid Loss')
    fig[0].set_title("Train and Valid Loss")
    if valid_iou_list is not None:
        fig[1].plot(valid_iou_list)
        fig[1].set_title("Valid IoU")
    if valid_dice_list is not None:
        fig[2].plot(valid_dice_list)
        fig[2].set_title("Valid Dice")
    plt.savefig(f'{results_folder}/train_result_fig.png')
    plt.show()
    
def dice_coefficient(loss):
    return 1 - loss
    
def dice_coefficient2(target, preds):
    temp = torch.zeros_like(preds[:, 1])
    temp[temp > 0.5] = 1
    preds = temp.unsqueeze(1)
    intersection = (preds * target).sum().float()
    set_sum = preds.sum() + target.sum()
    
    dice = (2 * intersection + 1e-8) / (set_sum + 1e-8) 
    
    return dice

def calculate_IoU(outputs, masks):
    predicted_masks = (outputs > 0.5).float()
    intersection = torch.sum(predicted_masks * masks)
    union = torch.sum(predicted_masks) + torch.sum(masks) - intersection
    iou = (intersection + 1e-8)/ (union + 1e-8)
    return iou
    
def visualize_dataset(train_loader):
    sam = iter(train_loader)
    nex = next(sam)
    print(nex[0].shape)
    print("Len of train_loader:", len(train_loader))
    ax, fig = plt.subplots(4, 10, figsize=(20, 10))
    for i in range(10):
        image1 = nex[0][i].squeeze().numpy()
        mask1 = nex[1][i].squeeze().numpy()
        image2 = nex[0][i+10].squeeze().numpy()
        mask2 = nex[1][i+10].squeeze().numpy()
        fig[0, i].imshow(image1, cmap='gray')
        fig[1, i].imshow(mask1, cmap='gray')
        fig[2, i].imshow(image2, cmap='gray')
        fig[3, i].imshow(mask2, cmap='gray')
    plt.show()

def dice_coefficient2_modified(target, preds):
    preds_flat = preds.view(-1)
    target_flat = target.view(-1)

    intersection = (preds_flat * target_flat).sum()
    set_sum = preds_flat.sum() + target_flat.sum()

    dice = (2 * intersection + 1e-8) / (set_sum + 1e-8) 

    return dice

def create_result_folder(path, with_time=True):
    # Get the current date and time
    current_datetime = datetime.now()
    if with_time is True:
        folder_name = current_datetime.strftime("%Y-%m-%d_%H-%M")
        new_path = os.path.join(path, folder_name)
        if os.path.exists(new_path) and os.path.isdir(new_path):
            pass
        else:
            os.makedirs(new_path)
        return new_path
    
def apply_gaussian_noise(img, std_dev):
    noise = torch.randn_like(img) * std_dev
    noisy_img = img + noise
    return noisy_img

def make_tfs(augs):
    return transforms.Compose([transforms.ToTensor()] + augs)

def custom_transformers(scale, contrast, brightness, rotation, blur, img_size=512):
    geometric_augs = [
        transforms.RandomResizedCrop(img_size, scale=scale) if scale else None,
        transforms.RandomRotation(rotation) if rotation else None,
        # transforms.GaussianBlur(blur) if blur else None,
    ]
    color_augs = [
        transforms.ColorJitter(brightness= brightness, contrast=contrast) if brightness and contrast else None,
    ]
    # tfs = transforms.Compose(geometric_augs)
    transform_input = make_tfs(geometric_augs + color_augs)
    transform_target = make_tfs(geometric_augs)
    return transform_input, transform_target

def custom_transformers_3d(scale, contrast, brightness, rotation, blur, volume_size=64):
    geometric_augs = [
        # transforms.RandomResizedCrop(volume_size, scale=(scale, scale)) if scale else None,
        transforms.RandomRotation(rotation) if rotation else None,
        # Note: GaussianBlur is not directly supported for 3D, you may need to implement a custom 3D blur function
    ]

    color_augs = [
        transforms.ColorJitter(brightness=brightness, contrast=contrast) if brightness and contrast else None,
    ]

    transform_input = make_tfs(geometric_augs + color_augs)
    transform_target = make_tfs(geometric_augs)
    return transform_input, transform_target



def train_fn(data_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        
        optimizer.zero_grad() 
        loss = criterion(outputs, masks)
            
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def show_image_modified(image, mask, pred_image=None, path_dir=None, filename=None, iou=None, dice_score=None):
    image = image.squeeze()  # Remove channel dimension if it's present
    mask = mask.squeeze()  # Remove channel dimension if it's present

    plt.figure(figsize=(10, 5))

    if pred_image is not None:
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
    else:
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

    ax1.set_title('IMAGE')
    ax1.imshow(image, cmap='gray')
    ax1.axis('off')

    ax2.set_title('GROUND TRUTH')
    ax2.imshow(mask, cmap='gray')
    ax2.axis('off')

    # Add IoU and Dice score to the predicted image subplot
    if pred_image is not None and iou is not None and dice_score is not None:
        pred_image = pred_image.squeeze()  # Remove channel dimension if it's present
        ax3.imshow(pred_image, cmap='gray')
        ax3.set_title('MODEL OUTPUT')
        ax3.axis('off')
        ax3.text(5, 5, f'IoU: {iou:.2f}, Dice: {dice_score:.2f}', color='white', fontsize=8, backgroundcolor='black')

    if path_dir is not None and filename is not None:
        full_path = os.path.join(path_dir, filename)
        plt.savefig(full_path, bbox_inches='tight', pad_inches=0)

    plt.close()

import numpy as np

def pad_or_crop_3d_numpy(image, mask, target_depth=512):
    # Get the original depth of the image
    original_depth = image.shape[0]

    # Calculate the padding or cropping needed
    padding_needed = max(0, target_depth - original_depth)
    cropping_needed = max(0, original_depth - target_depth)

    # Padding
    if padding_needed > 0:
        pad_lower = padding_needed // 2
        pad_upper = padding_needed - pad_lower
        image = np.pad(image, ((pad_lower, pad_upper), (0, 0), (0, 0)), mode='constant', constant_values=0)
        mask = np.pad(mask, ((pad_lower, pad_upper), (0, 0), (0, 0)), mode='constant', constant_values=0)
    # Cropping
    elif cropping_needed > 0:
        crop_lower = cropping_needed // 2
        crop_upper = cropping_needed - crop_lower
        image = image[crop_lower:original_depth - crop_upper, :, :]
        mask = mask[crop_lower:original_depth - crop_upper, :, :]

    return image, mask


def save_nifti(data, filename, affine=None):
    nifti_image = nib.Nifti1Image(data, affine)
    nib.save(nifti_image, filename)