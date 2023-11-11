import os
import torch
import numpy as np
import nibabel as nib
# from preprocess import Dataset2D, Dataset3D
from torch.utils.data import DataLoader
# from MyUNet import DeepUNet2D, UNet3D
from tqdm import tqdm
from torchmetrics.classification import Dice
# from preprocess import test_loader, test_image_paths, test_label_paths 
from collections import defaultdict
from monai.losses import DiceLoss
from torchvision import transforms
from utils import unet, mypreprocess, util_functions

MODE = '2d'
DEVICE = torch.device('cuda:0')
BEST_MODEL_PATH = '/scratch/student/sinaziaee/src/image_segmentation/results/best_model.pt'
OUTPUT_DIR = '/scratch/student/sinaziaee/src/image_segmentation/predictions'


def iou_score(target, preds):
    intersection = (preds * target).sum()
    union = preds.sum() + target.sum() - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.item()

dice_ce_loss = DiceLoss(to_onehot_y=True)

def load_model(mode, path):
    if mode == '2d':
        model = unet.UNet(num_classes=1, input_channels=1).to(DEVICE)
    elif mode == '3d':
        # model = UNet3D().to(DEVICE)
        pass
    
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def infer_and_save(model, loader, original_image_paths, dataset):
    running_iou = 0.0

    reconstructed_volume = defaultdict(list)
    probability_volume = defaultdict(list) 
    
    volume_idx_to_list_idx = {os.path.basename(path).split('_')[2]: i for i, path in enumerate(original_image_paths)}

    expected_slices_per_volume = defaultdict(int)
    for path in dataset.image_paths:
        volume_idx = os.path.basename(path).split('_')[2]
        expected_slices_per_volume[volume_idx] += 1

    with torch.no_grad():
        for i, (batch_images, batch_labels) in enumerate(tqdm(loader, desc="Inference", ncols=100)):
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            outputs = model(batch_images)
            
            for j in range(batch_images.size(0)): 
                preds = torch.sigmoid(outputs[j]) > 0.5
                iou_value = iou_score(batch_labels[j].cpu(), preds.cpu())
                running_iou += iou_value

                # Extract volume index from the filename
                filename = os.path.basename(dataset.image_paths[i * loader.batch_size + j])
                volume_idx = filename.split('_')[2]  

                # Extract the raw probability values
                slice_prob = outputs.cpu().numpy()[:, :, np.newaxis]

                # Threshold the probabilities to get the binary mask
                slice_mask = (slice_prob).astype(np.float32)

                reconstructed_volume[volume_idx].append(slice_mask)
                probability_volume[volume_idx].append(slice_prob)
                
                # Debugging print statement
                # print(f"Processed slice {j+1} for volume {volume_idx}. Total slices so far: {len(reconstructed_volume[volume_idx])}")

                if len(reconstructed_volume[volume_idx]) == expected_slices_per_volume[volume_idx]:
                    volume_3d = np.stack(reconstructed_volume[volume_idx], axis=2)
                    pred_nii = nib.Nifti1Image(volume_3d, affine=np.eye(4))

                    base_filename = os.path.basename(original_image_paths[volume_idx_to_list_idx[volume_idx]])
                    base_filename_parts = base_filename.split('_')
                    base_filename_without_slice = '_'.join(base_filename_parts[:3])

                    # Save the prediction
                    output_filename_pred = f"{base_filename_without_slice}_pred.nii.gz"
                    nib.save(pred_nii, os.path.join(OUTPUT_DIR, output_filename_pred))
                    print(f"Saved prediction for volume {volume_idx}")

                    # Save the probability map as .nii.gz
                    prob_volume_3d = np.stack(probability_volume[volume_idx], axis=2)
                    prob_nii = nib.Nifti1Image(prob_volume_3d, affine=np.eye(4))
                    output_filename_prob_txt = f"{base_filename_without_slice}_prob.txt"
                    np.savetxt(os.path.join(OUTPUT_DIR, output_filename_prob_txt), prob_nii, fmt='%f')
                    print(f"Saved probability map as txt for volume {volume_idx}")

                    del reconstructed_volume[volume_idx]
                    del probability_volume[volume_idx]
            
            del batch_images, batch_labels

    return running_iou / len(loader)

def get_slices_count(dataset):
    """Returns a list of slice counts for each 3D image in the dataset."""
    slice_counts = [0] * len(dataset.image_paths)
    for idx, slice_idx in dataset.index_map:
        slice_counts[idx] += 1
    return slice_counts

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    
    model = load_model(MODE, BEST_MODEL_PATH)
    base_path = '/scratch/student/sinaziaee/datasets/2d_dataset'
    test_dir = os.path.join(base_path, 'validation')
    img_size=512
    batch_size=64
    valid_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
    ])
    test_loader = mypreprocess.create_data_loaders(path_dir=test_dir, image_dir='images', 
                                            label_dir='labels', data_transformer=valid_transformer, batch_size=batch_size, split_size=None)
    
    avg_dice_score = infer_and_save(model, test_loader, test_loader.dataset.image_paths, test_loader.dataset)
    
    print(f"Average Dice Score for Inference: {avg_dice_score:.4f}")

if __name__ == "__main__":
    print(DEVICE)
    main()