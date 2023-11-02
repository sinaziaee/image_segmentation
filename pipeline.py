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
from utils import unet, mypreprocess
from tqdm import tqdm
from segmentation_models_pytorch.losses import DiceLoss

path_dir = os.path.join(os.getcwd(), 'Human-Segmentation-Dataset-master')
image_dir = os.path.join(path_dir, 'Training_Images')
label_dir = os.path.join(path_dir, 'Ground_Truth')
map_dir_path = os.path.join(path_dir, 'train.csv')
# print(image_dir, '-----', label_dir)
transform = mypreprocess.create_transformer(img_size=320)
train_loader, valid_loader, test_loader = mypreprocess.create_data_loaders(path_dir=path_dir, image_dir=image_dir, label_dir=label_dir, data_transformer=transform)

def train_fn(data_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        
        optimizer.zero_grad()
        # print(outputs.shape) 
        # print(masks.shape)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def eval_fn(data_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
    return avg_loss       

def visualize_training(train_loss_list, valid_loss_list, dice_score_list=None):
    ax, fig = plt.subplots(1, 2, figsize=(10, 4))
    fig[0].plot(train_loss_list)
    fig[1].plot(valid_loss_list)
    plt.savefig('results/train_result_fig.png')
    plt.show()
    
model = unet.UNet(num_classes=1, input_channels=1)
    
criterion1 = DiceLoss(mode="binary")
criterion2 = nn.BCEWithLogitsLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_epochs = 10
best_valid_loss = np.Inf

train_loss_list = []
valid_loss_list = []

for epoch in tqdm(range(n_epochs)):
    train_loss = train_fn(data_loader=train_loader, model=model, criterion=criterion1, optimizer=optimizer, device=device)
    valid_loss = eval_fn(data_loader=valid_loader, model=model, criterion=criterion1, device=device)
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    
    if best_valid_loss > valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'results/best_model.pt')
        print('SAVED-MODEL')
    
    # if epoch % 5 == 0:
    #     print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')
    
    visualize_training(train_loss_list=train_loss_list, valid_loss_list=valid_loss_list)
    