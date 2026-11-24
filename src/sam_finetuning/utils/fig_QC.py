"""QC figure of the first training batch (``SAM.module.visualize``)."""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# --------------------------------------------------------------------------- #

def visualize_input(image, gt_mask, box, pred_mask, label_id, image_name, model_save_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with three subplots

    # Original image
    # Assuming image is in [C, H, W] format and is an RGB image
    ax[0].imshow(image.permute(1, 2, 0), cmap='gray')  # Assuming image is in [C, H, W] format
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Ground truth mask
    ax[1].imshow(gt_mask, cmap='rainbow')
    ax[1].set_title('Ground Truth Mask')
    ax[1].axis('off')

    # Draw bounding boxes on the original image
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)

    # Predicted mask
    ax[2].imshow(pred_mask, cmap='rainbow')
    ax[2].set_title(f'Predicted Mask: {label_id}')
    ax[2].axis('off')

    figure_file_path = os.path.join(model_save_path, 'qual_check', f"{image_name}_input_viz.png")
    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

    plt.savefig(figure_file_path) 
    plt.close(fig)  # Close the figure to free memory
