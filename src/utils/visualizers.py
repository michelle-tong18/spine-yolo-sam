# Functions to get/load and display images

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

# -------------------- Color Tools --------------------
class ColorInfo():
    """
    TODO
    """
    # ---- FUNCTIONS NOT DEPENDENT ON DATA EXTENSION -----

    def set_unique_labels(self, mask):
        return [int(label) for label in np.unique(mask[mask != 0])]
    
    def set_unique_bbox_labels(self, volume_bbox_dict:dict):
        """
        - volume_bbox_dict: {int(slice): slice_bbox_list, ...}, where slice_bbox_list=['label_id bbox_pos bbox_pos bbox_pos bbox_pos', ...]
        """
        
        # Process each slice
        label_id_list = []
        for slice_idx, slice_bbox_list in volume_bbox_dict.items():
            # Process each label and get the bounding boxes
            for label_string in slice_bbox_list:
                label_id = int(label_string.split(" ", 1)[0])
                if label_id not in label_id_list:
                    label_id_list.append(label_id)

        return np.array(sorted(label_id_list))

    def set_image_clim(self, image):
        image_norm = 'percentile'
        # can do percentiles here
        if image_norm == 'percentile':
            return [np.percentile(image[:], 2),np.percentile(image[:], 98)]
        else:
            return [np.min(image[:]),np.max(image[:])]
    
    def set_mask_clim(self, mask):
        """
        Adjust 'clim' based on 'unique_labels' before calling 'adjust_cmap_for_labels'
        """
        unique_labels = self.set_unique_labels(mask)
        
        if not unique_labels:
            clim = [0,1]
        elif unique_labels == [0]:
            clim = [0,1]
        elif len(unique_labels) == 1:
            clim = [0, np.max(unique_labels)]
        elif len(unique_labels) > 1:
            clim = [min(unique_labels), max(unique_labels)]
        # didnt work for dhal or bacpac sag1
        return clim
    
    def set_bbox_colors(self, unique_labels, cmap:str='rainbow'):
        """
        Generate colors from the colormap
        - volume_bbox_dict: {int(slice): slice_bbox_list, ...}, where slice_bbox_list=['label_id bbox_pos bbox_pos bbox_pos bbox_pos', ...]
        
        """
        # Use the same adjust_cmap_for_labels logic as segmentation
        bbox_clim = [min(unique_labels), max(unique_labels)] if len(unique_labels) > 0 else [0, 1]
        adjusted_cmap, norm = self.adjust_cmap_for_labels(cmap, unique_labels, bbox_clim)

        # Get label colors from the adjusted colormap
        label_color_dict = {}
        for label in unique_labels:
            # Normalize label to [0, 1] range for colormap
            # normalized_value = (label - bbox_clim[0]) / (bbox_clim[1] - bbox_clim[0]) if bbox_clim[1] != bbox_clim[0] else 0.5
            label_color_dict[label] = adjusted_cmap(norm(label))
        return label_color_dict, adjusted_cmap, norm
                
        # # Get a color map
        # colormap = cm.get_cmap(cmap)
        # # Generate colors from the colormap
        # label_colors = [colormap(i / len(unique_labels)) for i in range(len(unique_labels))]
    
        # return dict(zip(unique_labels, label_colors))

    # Bug fix 12/3/25
    def adjust_cmap_for_labels(self, cmap, labels, clim):
        """
        Adjust a given colormap to fit the specified labels within the color limit (clim) range,
        ensuring each label has a consistent color across the volume.
        """
        labels = sorted(labels)
        base = plt.get_cmap(cmap)
        if not labels:
            # All-background seg: no discrete bins; avoid labels[-1] on empty list.
            return base, Normalize(0, 1)

        vmin, vmax = clim

        # map each label to a fractional position between 0 and 1
        # relative to the clim range
        frac = [(label - vmin) / (vmax - vmin) for label in labels]
        frac = np.clip(frac, 0, 1)

        # sample the base colormap at these positions
        colors = base(frac)
        new_cmap = ListedColormap(colors, name=f"{cmap}_discrete")

        # make boundaries halfway between successive label values (plus margins)
        bounds = []
        for i, lab in enumerate(labels):
            if i == 0:
                bounds.append(lab - 0.5)
            else:
                bounds.append((labels[i - 1] + lab) / 2)
        bounds.append(labels[-1] + 0.5)

        norm = BoundaryNorm(bounds, new_cmap.N, clip=True)
        return new_cmap, norm
    
    def add_colorbar(self, fig, ax, labels_dict, cmap, norm, unique_labels):
        # Assuming background is already excluded from unique_labels and labels_dict

        # Calculate the boundaries and ticks using the unique labels
        # boundaries = np.linspace(norm.vmin, norm.vmax, len(unique_labels) + 1)
        boundaries = []
        for i, lab in enumerate(unique_labels):
            if i == 0:
                boundaries.append(lab - 0.5)
            else:
                boundaries.append((unique_labels[i - 1] + lab) / 2)
        boundaries.append(unique_labels[-1] + 0.5)

        # tick positions: midpoints of each color bin
        tick_positions = [(boundaries[i] + boundaries[i+1]) / 2
                        for i in range(len(boundaries) - 1)]

        # Create the colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for ScalarMappable
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', #fraction=0.046, pad=0.04, 
                            boundaries=boundaries, # important
                            ticks=tick_positions)

        # Set the tick labels
        tick_labels = [labels_dict[label] for label in unique_labels]
        cbar.set_ticklabels(tick_labels)

        cbar.ax.tick_params(labelsize=12)
        plt.setp(cbar.ax.get_xticklabels(), rotation=45, ha='right')


# -------------------- Plot Images (2D) --------------------
# import cv2

# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([251/255, 252/255, 30/255, 0.5])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)

# def save_slice_image(slice_img, filepath):
#     """
#     Saves a slice image to the specified filepath.
#     """
#     plt.imshow(slice_img, cmap='gray')
#     plt.axis('off')
#     plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
#     plt.close()


# -------------------- Plot Volumes (3D) --------------------
        


# -------------------- Plot Segmentation Overlays --------------------
    
def plot_segmentation_overlay(vol, seg, save_path:str=None, cols:int=6, scale:int=3, \
        seg_clim=None, cmap='rainbow', title:str=None, labels_dict={}): 
    """
    Plot volume slices with segmentation overlay and optionally save to a file.
    Parameters:
    - vol: The volume to be displayed.
    - seg: The segmentation mask to overlay.
    - cols: Number of columns in the subplot grid.
    - scale: Scaling factor for each subplot.
    - clim: Color limits for the segmentation overlay.
    - cmap: Colormap for the segmentation overlay ('rainbow' by default).
    - title: Title for the plot.
    - save_path: Path to save the figure.
    - labels_dict: Specifies semantic labels (values) associated with segmentation array (keys). 
        Ex. {0: background, 1: anatomy} 
    """
    #Define properties for plot colors
    ColorFns = ColorInfo()
    
    image_clim = ColorFns.set_image_clim(vol)
    #if mask_clim is None:
    seg_clim = ColorFns.set_mask_clim(seg)
    unique_labels = ColorFns.set_unique_labels(seg)
    adjusted_cmap, norm = ColorFns.adjust_cmap_for_labels(cmap, unique_labels, seg_clim)
    
    num_images = vol.shape[0]
    if num_images <= 0:
        return
    rows = (num_images - 1) // cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * scale, rows * scale))
    axes = np.atleast_1d(axes).flatten()
    
    for idx in range(num_images):
        ax = axes[idx]
        ax.imshow(np.squeeze(vol[idx]), cmap='gray', clim=image_clim, aspect='equal')
        overlay = np.ma.masked_where(seg[idx] == 0, seg[idx])
        im = ax.imshow(overlay, cmap=adjusted_cmap, norm=norm, alpha=0.4, clim=seg_clim)
        ax.axis('off')
    
    # Adjust for empty subplots if num_images is not a multiple of cols
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    if title:
        plt.suptitle(title, fontsize=20)
    
    if labels_dict and (unique_labels != []) and (num_images > 0):
        ColorFns.add_colorbar(fig, axes, labels_dict, adjusted_cmap, norm, unique_labels)
    
    # Save
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return
    else:
        return fig


# -------------------- Plot Segmentation Contours --------------------

    

# -------------------- Plot Bbox Overlays --------------------

def yolo_to_xyxy(yolo_bbox, img_width, img_height):
    """
    Convert YOLO format bbox to corner format.
    
    Parameters:
    - yolo_bbox: [x_center, y_center, width, height] in normalized coordinates (0-1)
    - img_width: image width in pixels
    - img_height: image height in pixels
    
    Returns:
    - [x_min, y_min, x_max, y_max] in pixel coordinates
    """
    x_center, y_center, width, height = yolo_bbox
    
    # Convert from normalized to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Convert to corner format
    x_min = x_center_px - width_px / 2
    y_min = y_center_px - height_px / 2
    x_max = x_center_px + width_px / 2
    y_max = y_center_px + height_px / 2
    
    return [x_min, y_min, x_max, y_max]

def show_box(box, ax, edgecolor='blue'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2))

def plot_bbox_overlay(vol, volume_bbox_dict, save_path:str=None, cols:int=6, scale:int=3, \
        cmap='rainbow', title:str=None, labels_dict=None): 
    """
    Plot volume slices with segmentation overlay and optionally save to a file.
    Parameters:
    - vol: The volume to be displayed.
    - volume_bbox_dict: {int(vol_slice_idx): [str('label, position, pos, pos, pos')]}
    - cols: Number of columns in the subplot grid.
    - scale: Scaling factor for each subplot.
    - clim: Color limits for the segmentation overlay.
    - cmap: Colormap for the segmentation overlay ('rainbow' by default).
    - title: Title for the plot.
    - save_path: Path to save the figure.
    - labels_dict: Specifies semantic labels (values) associated with segmentation array (keys). 
        Ex. {0: background, 1: anatomy} 
    """

    #Define properties for plot colors
    ColorFns = ColorInfo()
    image_clim = ColorFns.set_image_clim(vol)
    unique_labels = ColorFns.set_unique_bbox_labels(volume_bbox_dict)
    label_color_dict, adjusted_cmap, norm = ColorFns.set_bbox_colors(unique_labels, cmap)

    num_images = len(volume_bbox_dict.keys())
    rows = (num_images - 1) // cols + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * scale, rows * scale))
    axes = axes.flatten()

    # Draw each bounding box with a color from the colormap
    for plt_idx, slice_idx in enumerate(volume_bbox_dict.keys()):
        ax = axes[plt_idx]
        ax.imshow(np.squeeze(vol[int(slice_idx)]), cmap='gray', clim=image_clim, aspect='equal')

        slice_bbox_list = volume_bbox_dict[slice_idx]
        for label_str in slice_bbox_list:
            parts = label_str.split()
            label_id = int(parts[0])
            yolo_bbox = [float(x) for x in parts[1:]]
            # Convert YOLO format to corner format
            bbox_xyxy = yolo_to_xyxy(yolo_bbox, vol.shape[1], vol.shape[2])
            show_box(bbox_xyxy, ax, edgecolor=label_color_dict[label_id])
        ax.axis('off')

    # Adjust for empty subplots if num_images is not a multiple of cols
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    if title:
        plt.suptitle(title, fontsize=20)
    
    # Add colorbar if labels_dict is provided
    if labels_dict:
        ColorFns.add_colorbar(fig, axes, labels_dict, adjusted_cmap, norm, unique_labels)

    # Save
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return
    else:
        return fig


