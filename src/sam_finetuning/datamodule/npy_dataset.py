"""SAM finetuning dataset: one (image, mask of one label, box) sample per preprocessed slice."""
import random
import numpy as np

import torch
from torch.utils.data import Dataset

from scipy.ndimage import label as scipy_label

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from .components import build_augmentation_pipeline

# ----------------- Dataset for Finetuning ----------------- #
class SAMDataset(Dataset):
    """ Finetuning scripts now in development for instance-based finetuning """
    def __init__(self, gt_paths, img_paths, bbox_shift=0, instance_bbox=False, remove_label_ids=[], dataset_name=None, augmentation_config=None):

        self.gt_path_files = gt_paths
        self.img_path_files = img_paths
        self.bbox_shift = bbox_shift
        self.instance_bbox = instance_bbox
        self.remove_label_ids = remove_label_ids
        self.datset_name = dataset_name

        # Use the imported function to build the pipeline
        self.augmentation_pipeline = build_augmentation_pipeline(augmentation_config)

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):

        attempt = 0
        max_attempts = len(self.gt_path_files)

        while attempt < max_attempts:
            attempt += 1
                
            # Load npy image (1024, 1024), [0,1] 
            img_name = self.img_path_files[index]
            try:
                img_1024 = np.load(img_name, mmap_mode='r')  # img_1024.shape -> (1024, 1024, 3)
            except (ValueError, OSError) as e:
                print(f'Image {img_name} failed to load ({e}). Skipping to the next image.')
                index = (index + 1) % len(self.gt_path_files)
                continue

            if img_1024.ndim == 2 or (img_1024.ndim == 3 and img_1024.shape[2] == 1):
                img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)  # (1024, 1024, 3)

            # img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)  # (1024, 1024, 3)

            # Convert the shape to (3, H, W)
            img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)
            
            # Check the minimum and maximum values
            min_val, max_val = img_1024.min(), img_1024.max()

            # If values are outside the range [0, 1], continue to the next iteration
            if min_val < 0 or max_val > 1:
                print(f'Image {img_name} is out of 0-1 range [{min_val}, {max_val}]. Skipping to the next image.')
                index = (index + 1) % len(self.gt_path_files)
                continue

            # Load npy mask (1024, 1024)
            try:
                gt = np.load(self.gt_path_files[index], mmap_mode='r')
            except (ValueError, OSError) as e:
                print(f'Mask {self.gt_path_files[index]} failed to load ({e}). Skipping to the next image.')
                index = (index + 1) % len(self.gt_path_files)
                continue

            label_ids = np.unique(gt)[1:]  # Exclude background (0)

            # Exclude labels as specified in remove_label_ids
            label_ids = [label for label in label_ids if label not in self.remove_label_ids]

            # Handling the case where all labels are excluded
            if not label_ids:
                raise ValueError(f"All labels have been removed for image at index {index}.")

            chosen_label = random.choice(label_ids)

            gt2D = np.uint8(gt == chosen_label)  # Binary mask for chosen class

            if self.augmentation_pipeline is not None:
                augmented = self.augmentation_pipeline(image=img_1024, mask=gt2D)
                img_1024 = augmented['image']
                gt2D = augmented['mask']

            if self.instance_bbox:
                # Find connected components for the current label
                labeled_array, num_features = scipy_label(gt2D)

                # Randomly choose one of these instances
                chosen_component = random.choice(range(1, num_features + 1))
                component_mask = labeled_array == chosen_component
                
                y_indices, x_indices = np.where(component_mask)

                # Compute the bounding box for the selected component
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # Add perturbation to bounding box coordinates, similar to your original approach
                H, W = gt2D.shape
                x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
                y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

                bboxes = np.array([x_min, y_min, x_max, y_max])
        
                # Update gt2D to only include the selected component
                gt2D = component_mask.astype(np.uint8)
                
            else:
                # Compute bounding box from the augmented single-class mask
                y_indices, x_indices = np.where(gt2D > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # add perturbation to bounding box coordinates
                H, W = gt2D.shape
                x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
                y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

                bboxes = np.array([x_min, y_min, x_max, y_max])
                
            batch_data = {
                'image': torch.tensor(img_1024).float(),
                'gt2D': torch.tensor(gt2D[None, :, :]).long(),
                'boxes': torch.tensor(bboxes).float(),
                'label_id': torch.tensor(chosen_label).long(),
                'img_name': img_name,
                'dataset_name': self.datset_name,
            }

            return batch_data
        
        # If no valid data found after max_attempts
        raise ValueError("Could not find a valid image after multiple attempts.")
