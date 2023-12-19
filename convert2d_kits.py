import os
import nibabel as nib
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
import numpy as np
import imageio
import logging

qualified_images_count = 0
non_conforming_images = []

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def gather_paths(data_folder):
    image_paths = []
    label_paths = []
    
    for case_folder in sorted(os.listdir(data_folder)):
        case_path = os.path.join(data_folder, case_folder)
        if os.path.isdir(case_path):
            image_path = os.path.join(case_path, 'imaging.nii.gz')
            label_path = os.path.join(case_path, 'segmentation.nii.gz')
            
            if os.path.exists(image_path) and os.path.exists(label_path):
                image_paths.append(image_path)
                label_paths.append(label_path)
    
    return image_paths, label_paths

def split_data(image_paths, label_paths, validation_ratio=0.2):
    paired_paths = list(zip(image_paths, label_paths))
    train_pairs, val_pairs = train_test_split(paired_paths, test_size=validation_ratio, random_state=42)
    train_image_paths, train_label_paths = zip(*train_pairs)
    val_image_paths, val_label_paths = zip(*val_pairs)
    return (train_image_paths, train_label_paths), (val_image_paths, val_label_paths)

def pad_or_crop_3d_image(image_3d, target_size=(512, 512, 512)):
    padding = [(target_size[i] - image_3d.shape[i]) // 2 for i in range(3)]
    if any(p > 0 for p in padding):
        padded_image = np.pad(
            image_3d,
            (
                (max(padding[0], 0), max(padding[0], 0)),
                (max(padding[1], 0), max(padding[1], 0)),
                (max(padding[2], 0), max(padding[2], 0))
            ),
            mode='constant',
            constant_values=0
        )
        return padded_image
    else:
        crop_x = (image_3d.shape[0] - target_size[0]) // 2
        crop_y = (image_3d.shape[1] - target_size[1]) // 2
        crop_z = (image_3d.shape[2] - target_size[2]) // 2
        cropped_image = image_3d[
            crop_x:crop_x + target_size[0],
            crop_y:crop_y + target_size[1],
            crop_z:crop_z + target_size[2]
        ]
        return cropped_image

def transform_image(image_3d):
    # Flip the image along the y-axis and then along the z-axis
    flipped_y = np.flip(image_3d, axis=1)
    flipped_z = np.flip(flipped_y, axis=2)
    return flipped_z

def save_slice(img_slice, lbl_slice, img_slice_path, lbl_slice_path):
    # Rotate the slice by 90 degrees to the left
    rotated_img_slice = np.rot90(img_slice, k=-1, axes=(0, 1))
    rotated_lbl_slice = np.rot90(lbl_slice, k=-1, axes=(0, 1))

    img_slice_uint8 = (255 * (rotated_img_slice - rotated_img_slice.min()) / (rotated_img_slice.max() - rotated_img_slice.min())).astype(np.uint8)
    lbl_slice_uint8 = (255 * (rotated_lbl_slice - rotated_lbl_slice.min()) / (rotated_lbl_slice.max() - rotated_lbl_slice.min())).astype(np.uint8)

    imageio.imwrite(img_slice_path, img_slice_uint8)
    imageio.imwrite(lbl_slice_path, lbl_slice_uint8)

def save_2d_slices(paths, output_folder, set_type):
    global qualified_images_count, non_conforming_images

    for img_index, (img_path, lbl_path) in enumerate(paths):
        try:
            # Load and pad/crop the images
            img_3d_nii = nib.load(img_path)
            lbl_3d_nii = nib.load(lbl_path)

            img_3d = img_3d_nii.get_fdata()
            lbl_3d = lbl_3d_nii.get_fdata()

            img_3d_padded_or_cropped = pad_or_crop_3d_image(img_3d)
            lbl_3d_padded_or_cropped = pad_or_crop_3d_image(lbl_3d)

            # Transform the images
            img_3d_transformed = transform_image(img_3d_padded_or_cropped)
            lbl_3d_transformed = transform_image(lbl_3d_padded_or_cropped)

            depth = img_3d_padded_or_cropped.shape[2]

            if img_3d_padded_or_cropped.shape[0] == 512 and img_3d_padded_or_cropped.shape[1] == 512:
                qualified_images_count += 1
            else:
                non_conforming_images.append(os.path.basename(img_path))

            img_output_folder = os.path.join(output_folder, set_type, 'images')
            lbl_output_folder = os.path.join(output_folder, set_type, 'labels')
            os.makedirs(img_output_folder, exist_ok=True)
            os.makedirs(lbl_output_folder, exist_ok=True)

            case_folder = os.path.basename(os.path.dirname(img_path))
            case_id = case_folder.split('_')[1]

            for slice_idx in range(depth):
                img_slice = img_3d_padded_or_cropped[slice_idx, :, :]
                lbl_slice = lbl_3d_padded_or_cropped[slice_idx, :, :]

                img_filename = f"{set_type}_image_{case_id}_{str(slice_idx).zfill(4)}.png"
                lbl_filename = f"{set_type}_label_{case_id}_{str(slice_idx).zfill(4)}.png"

                img_slice_path = os.path.join(img_output_folder, img_filename)
                lbl_slice_path = os.path.join(lbl_output_folder, lbl_filename)

                save_slice(img_slice, lbl_slice, img_slice_path, lbl_slice_path)

                logging.info(f"Saved: {img_slice_path}, {lbl_slice_path}")

        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")

# DATASET_FOLDER = "/work/ovens_lab/thaonguyen/uncertainty/kits23/dataset"
# OUTPUT_FOLDER = "/work/ovens_lab/thaonguyen/image_segmentation/kits_2d_dataset_new"
DATASET_FOLDER = "/scratch/student/sinaziaee/kits23/dataset"
OUTPUT_FOLDER = "/scratch/student/sinaziaee/datasets/kits_2d_dataset_new"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

all_image_paths, all_label_paths = gather_paths(DATASET_FOLDER)
(train_image_paths, train_label_paths), (val_image_paths, val_label_paths) = split_data(all_image_paths, all_label_paths)

save_2d_slices(list(zip(train_image_paths, train_label_paths)), OUTPUT_FOLDER, 'training')

save_2d_slices(list(zip(val_image_paths, val_label_paths)), OUTPUT_FOLDER, 'validation')

test_image_paths, test_label_paths = gather_paths(DATASET_FOLDER) 
save_2d_slices(list(zip(test_image_paths, test_label_paths)), OUTPUT_FOLDER, 'testing')

print(f"Number of qualified images (512x512): {qualified_images_count}")

if non_conforming_images:
    print(f"Number of non-conforming images (not 512x512): {len(non_conforming_images)}")
    print(f"List of indexes: {non_conforming_images}")
else:
    print("All images conform to the expected dimensions.")