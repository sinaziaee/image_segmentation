import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime

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
        # if path_dir is not None:
        #     plt.savefig(f'{path_dir}/{num}.png')


def visualize_training(train_loss_list, valid_loss_list, valid_iou_list=None, dice_score_list=None, valid_dice_list=None, results_folder= None):
    ax, fig = plt.subplots(1, 4, figsize=(16, 4))
    fig[0].plot(train_loss_list)
    fig[0].set_title("Train Loss")
    fig[1].plot(valid_loss_list)
    fig[1].set_title("Valid Loss")
    if valid_iou_list is not None:
        fig[2].plot(valid_iou_list)
        fig[2].set_title("Valid IoU")
    if valid_dice_list is not None:
        fig[3].plot(valid_dice_list)
        fig[3].set_title("Valid Dice")
    plt.savefig(f'{results_folder}/train_result_fig.png')
    plt.show()
    
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
    iou = intersection / union
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