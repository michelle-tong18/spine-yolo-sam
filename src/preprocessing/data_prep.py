import math
import warnings

import cv2
import numpy as np
from skimage import transform

from src.preprocessing.data_loaders.dataset_handler import get_data
from src.preprocessing.data_transforms import transform_data, crop_data_to_slices, find_non_zero_slices, validate_data, extrapolate_z

# ---------------------- UTILITY FUNCTIONS ----------------------

class DataPrep():
    """
    Class of static variables and functions to prepare data for YOLO & SAM training or inference. 
    This is a base class image prep and mask prep. Supports 2D or 3D input.
    
    Static Params:
    - image_size_tuple: (H, W) in pixels to resize mask using nearest-neighbor interpolation. This 
        will preserve image intensities.
    - make_square: If True, will pad the mask to make it square before resizing.
    - ratio_resize: If True, will resize the mask while maintaining the original aspect ratio.
    - data_transforms: A dictionary of data transformations to apply.
    - verbose: If True, will print out information during processing.
    """

    def __init__(self, image_size_tuple:tuple[int, int] = [], make_square:bool=False, ratio_resize:bool=False, data_transforms:dict={}, slice_normalization:bool=False, verbose=False):
        self.image_size_tuple = image_size_tuple
        self.make_square = make_square
        self.ratio_resize = ratio_resize
        self.data_transforms = data_transforms
        self.slice_normalization = slice_normalization
        self.verbose = verbose

    def get_data(self, filepath: str, no_dicom_ext_flag: bool = False) -> tuple[np.ndarray, tuple]:
        """(data after ``data_transforms``, shape as loaded) of one scan (src/preprocessing/data_loaders)."""
        return get_data(filepath, data_transforms=self.data_transforms, no_dicom_ext_flag=no_dicom_ext_flag)

    def data_dims(self, data: np.ndarray) -> int:
        """Return the number of dimensions of the input data."""
        return len(np.shape(data))

    def resize_data_2D(self, img_slice, image_size_tuple:(int,int), order:int=0, anti_alias:bool=True):
        # Resize the normalized image slice
        resized_img = transform.resize(
            img_slice,
            image_size_tuple,
            order=order,  # nn or cubic spline interpolation
            preserve_range=True,
            mode='constant',
            anti_aliasing=anti_alias
        )
        return resized_img

    def scaled_resize_data_2D(self, img_slice, image_size_tuple:(int,int), order:int=0):
        if order==0:
            interpolation=cv2.INTER_NEAREST
        elif order==1:
            interpolation=cv2.INTER_LINEAR
        elif order==2:
            interpolation=cv2.INTER_AREA
        elif order==3:
            interpolation=cv2.INTER_CUBIC
        else:
            interpolation=cv2.INTER_LINEAR

        h0, w0 = img_slice.shape[:2]  # Original dimensions
        # Resize long side to imgsz while maintaining aspect ratio
        r = image_size_tuple[0]/ max(h0, w0)  # Ratio
        if r != 1:  # If sizes are not equal
            w, h = (min(math.ceil(w0 * r), image_size_tuple[0]), min(math.ceil(h0 * r), image_size_tuple[0]))
            resized_img = cv2.resize(img_slice, (w, h), interpolation=interpolation)  # cv2.INTER_LINEAR
            return resized_img
        return img_slice
    
    def pad_to_square(self, img_slice):
        height, width = img_slice.shape
        if height > width:
            pad = (height - width) // 2
            padded_mask = np.pad(img_slice, ((0, 0), (pad, height - width - pad)), mode='constant')
        elif width > height:
            pad = (width - height) // 2
            padded_mask = np.pad(img_slice, ((pad, width - height - pad), (0, 0)), mode='constant')
        else:
            padded_mask = img_slice  # already square
        return padded_mask

    def unpad_to_original_size(self, img_slice, original_shape):
        if len(original_shape) == 2:
            height, width = original_shape
        elif len(original_shape) == 3:
            _, height, width = original_shape
        else:
            raise ValueError("Unsupported original_shape dimensions")

        if height > width:
            pad = (height - width) // 2
            unpadded_mask = img_slice[:, pad:pad + width]
        elif width > height:
            pad = (width - height) // 2
            unpadded_mask = img_slice[pad:pad + height, :]
        else:
            unpadded_mask = img_slice  # already square
        return unpadded_mask

class ImagePrep(DataPrep):
    """
    Class of static variables and functions to prepare images for SAM/YOLO. Supports 2D or 3D image input.
    
    Static Params:
    - image_size_tuple: (H, W) in pixels to resize mask using nearest-neighbor interpolation. This 
        will preserve image intensities.
    - make_square: If True, will pad the mask to make it square before resizing.
    - ratio_resize: If True, will resize the mask while maintaining the original aspect ratio.
    - verbose: If True, will print out information during processing.

    Input Params:
    - image_data: The image data as a numpy array. Supports 2D slices or 3D volumes.
    - z_indices: True or False. Indicate whether to crop volume to only slices specified. This makes it
        possible to only select slices with segmentations. (Only possible for 3D masks)
    """
    def __init__(self, image_size_tuple:tuple[int, int] = [], make_square:bool=False, ratio_resize:bool=False, data_transforms:dict={}, slice_normalization:bool=False, verbose=False):
        super().__init__(image_size_tuple=image_size_tuple, make_square=make_square, ratio_resize=ratio_resize, data_transforms=data_transforms, slice_normalization=slice_normalization, verbose=verbose)

    def resize_images(self, image_data: np.ndarray, image_size_tuple:tuple[int, int]=None, n_channels:int=0, order:int=3) -> np.ndarray:
        """Resize image data using cubic spline interpolation."""
        
        if image_size_tuple is None:
            image_size_tuple = self.image_size_tuple

        # original_dtype = image_data.dtype

        if self.ratio_resize:
            dims = self.data_dims(image_data)
            if (dims==2 and n_channels==0) or (dims==3 and n_channels>0):   
                resized_images = self.scaled_resize_data_2D(image_data, image_size_tuple, order=order)
            elif (dims==3 and n_channels==0) or (dims==4 and n_channels>0):
                resized_images = []
                for img_slice in image_data:
                    resized_img = self.scaled_resize_data_2D(img_slice, image_size_tuple, order=order)
                    resized_images.append(resized_img)
                resized_images = np.array(resized_images)

        else:

            if self.make_square:
                if len(image_data.shape) == 2:
                    image_data = self.pad_to_square(image_data)
                elif len(image_data.shape) == 3 and n_channels == 0:
                    image_data = np.array([self.pad_to_square(slice) for slice in image_data])
                elif len(image_data.shape) == 3 and n_channels > 0:
                    image_data = np.array([self.pad_to_square(slice) for slice in image_data])
                elif len(image_data.shape) == 4 and n_channels > 0:
                    image_data = np.array([[self.pad_to_square(slice) for slice in img_channel] for img_channel in image_data])

            dims = self.data_dims(image_data)
            if (dims==2 and n_channels==0) or (dims==3 and n_channels>0):   
                resized_images = self.resize_data_2D(image_data, image_size_tuple, order=order)
            elif (dims==3 and n_channels==0) or (dims==4 and n_channels>0):
                resized_images = []
                for img_slice in image_data:
                    resized_img = self.resize_data_2D(img_slice, image_size_tuple, order=order)
                    resized_images.append(resized_img)
                resized_images = np.array(resized_images)
        
        # resized_images = resized_images.astype(original_dtype)

        return resized_images

    def normalize_image_data_by_slice(self, image_data: np.ndarray) -> np.ndarray:
        """Normalize image data by each slice so that intensities range from 0 to 1."""

        if self.data_dims(image_data) == 2:
            normed_images = transform_data(image_data, {'normalize': 'data_max'})
        elif self.data_dims(image_data) == 3:
            normed_images = []
            for img_slice in image_data:
                normed_img = transform_data(img_slice, {'normalize': 'data_max'})
                normed_images.append(normed_img)
            normed_images = np.array(normed_images)
        return normed_images
    
    def prep_image(self, image_data: np.ndarray, z_indices:list[int] = [], original_z: list[int] = []) -> np.ndarray:
        """
        Prepare image data for SAM/YOLO training or inference, including normalization and resizing.
        Args:
        - image_data: Supports 2D [H x W] and 3D [Slice x H x W] image data.
        - z_indices: relevant slices to keep if image_data is 3D
        Returns:
            - Prepared image data as a numpy array.
        """

        assert self.data_dims(image_data) == 2 or self.data_dims(image_data) == 3, "Image data must be 2D, or 3D numpy array."

        original_dtype = image_data.dtype

        if self.verbose:
            print(f"Image data shape: {np.shape(image_data)}")
            print(f"Image data range: {np.min(image_data)} - {np.max(image_data)}")
            print(f"Valid pixels >0: {np.sum(image_data>0)}")
            print(f"Image data dtype: {original_dtype}")
            print(f"Z-indices for cropping: {z_indices}")
        if len(z_indices):
            # transform z-indices if mask size differs from image size
            if len(original_z) > 0:
                if np.shape(image_data)[0] != len(original_z): #modified from np.shape(image_data [0]) May 2026
                    z_indices = extrapolate_z(z_indices, np.shape(image_data)[0], len(original_z))

                if self.verbose:
                    print(f"Original z shape: {original_z}")
                    print(f"Transformed z-indices for cropping: {z_indices}")
            # Cropping the image data slices to match the cropped mask data
            image_data = crop_data_to_slices(image_data, z_indices)
            # Ensure image has non-zero intensities
            if np.sum(image_data!=0) == 0:
                raise ValueError('No non-zero intensities in image.')
            
        if self.verbose:
            print(f"Cropped image data shape: {np.shape(image_data)}")
            print(f"Cropped image data range: {np.min(image_data)} - {np.max(image_data)}")
            print(f"Cropped valid pixels >0: {np.sum(image_data>0)}")

        # normalize data to range 0 - 1
        normalized_data = transform_data(image_data, {'normalize': 1})

        # Apply slice-wise normalization if specified
        if self.slice_normalization:
            normalized_data = self.normalize_image_data_by_slice(normalized_data)

        if self.verbose:
            print(f"Normalized image data shape: {np.shape(normalized_data)}")
            print(f"Normalized image data range: {np.min(normalized_data)} - {np.max(normalized_data)}")

        # Resize and image data
        resized_normalized_data = self.resize_images(normalized_data)

        # Convert back to original data type
        resized_normalized_data = transform_data(resized_normalized_data, {'normalize': 'np.'+str(original_dtype)})
        resized_normalized_data = transform_data(resized_normalized_data, {'dtype': 'np.'+str(original_dtype)})

        if self.verbose:
            print(f"Final Image data dtype: {resized_normalized_data.dtype}")
            print(f"Final Image data shape: {np.shape(resized_normalized_data)}")
            print(f"Final Image data range: {np.min(resized_normalized_data)} - {np.max(resized_normalized_data)}")
            print(f"Final Valid pixels >0: {np.sum(resized_normalized_data>0)}")

        return resized_normalized_data

    def get_prepped_data(self, filepath: str, no_dicom_ext_flag: bool = False, z_indices: list[int] = [],
                         original_z: list[int] = []) -> tuple[np.ndarray, tuple]:
        """Load and prepare one image volume (``prep_image``); returns (prepared image, shape as loaded)."""
        image_data, original_image_shape = self.get_data(filepath, no_dicom_ext_flag=no_dicom_ext_flag)
        return self.prep_image(image_data, z_indices=z_indices, original_z=original_z), original_image_shape

    def validate_image(self, data: np.ndarray, expected_properties: dict):
        """
        Validate the data against expected properties.
        Args:
            data: The data to validate.
            expected_properties: A dictionary of expected properties to check against.
        """
        if expected_properties:
            validate_data(data, expected_properties, content='image')

class MaskPrep(DataPrep):
    """
    Prepare label masks for SAM/YOLO (2D or 3D): drop ``remove_label_ids``, remove small objects, crop to labelled
    slices (plus an optional random ``z_sample`` share of the others), and resize with nearest neighbour.

    Static Params:
    - image_size_tuple: (H, W) in pixels to resize mask using nearest-neighbor interpolation.
    - make_square: If True, will pad the mask to make it square before resizing.
    - ratio_resize: If True, will resize the mask while maintaining the original aspect ratio.
    - verbose: If True, will print out information during processing.
    """
    def __init__(self, remove_label_ids: list = [], voxel_threshold_3d: int = 0, pixel_threshold_2d: int = 0,
                 crop_non_zero_slices_flag: bool = True, z_sample: float = 0.0, image_size_tuple: tuple[int, int] = [],
                 make_square: bool = False, ratio_resize: bool = False, data_transforms: dict = {}, verbose=False):
        super().__init__(image_size_tuple=image_size_tuple, make_square=make_square, ratio_resize=ratio_resize, data_transforms=data_transforms, verbose=verbose)
        self.remove_label_ids = remove_label_ids
        self.voxel_threshold_3d = voxel_threshold_3d
        self.pixel_threshold_2d = pixel_threshold_2d
        self.crop_non_zero_slices_flag = crop_non_zero_slices_flag
        self.z_sample = z_sample or 0.0

    def remove_labels(self, mask_data: np.ndarray) -> np.ndarray:
        """Set the ``remove_label_ids`` labels to background."""
        if not self.remove_label_ids:
            return mask_data
        if not isinstance(self.remove_label_ids, list):
            raise TypeError("remove_label_ids must be a list.")
        mask_data[np.isin(mask_data, self.remove_label_ids)] = 0
        return mask_data

    def remove_small_objects(self, mask_data: np.ndarray) -> np.ndarray:
        """Remove connected components below ``voxel_threshold_3d`` (3D) and ``pixel_threshold_2d`` (per slice)."""
        if self.data_dims(mask_data) == 2:
            return transform_data(mask_data, {'remove_small_cc_2D': self.pixel_threshold_2d})
        mask_data = transform_data(mask_data, {'remove_small_cc_3D': self.voxel_threshold_3d})
        return transform_data(mask_data, {'remove_small_cc_2D': self.pixel_threshold_2d})

    def multidim_mask_func(self, mask_data, dims, func, *args):
        """Apply a 2D function to a 2D mask, or to every slice of a 3D mask."""
        if dims == 2:
            return func(mask_data, *args)
        if dims == 3:
            return np.array([func(mask_slice, *args) for mask_slice in mask_data])
        raise ValueError("Unsupported number of dimensions: {}".format(dims))

    def resize_mask(self, mask_data, image_size_tuple:tuple[int,int]= None, order=0):
        """
        Resize mask data using nearest-neighbor interpolation to preserve label integrity.
        Parameters:
        - mask_data: The mask data as a numpy array.
        - image_size_tuple: (H, W) in pixels to resize mask.

        Returns:
        - A numpy array with dimensions (H, W).
        """
        if image_size_tuple is None:
            image_size_tuple = self.image_size_tuple
        mask_shape = np.shape(mask_data)
        data_dims = self.data_dims(mask_data)
        original_shape = mask_shape[1:] if data_dims == 3 else mask_shape

        if self.ratio_resize:
            return self.multidim_mask_func(mask_data, data_dims, self.scaled_resize_data_2D, image_size_tuple, order)
        if self.make_square and ((original_shape[0] != original_shape[1]) or (image_size_tuple[0] != image_size_tuple[1])):
            if (original_shape[0] > image_size_tuple[0]) or (original_shape[1] > image_size_tuple[1]):
                adjusted_size = (np.max(image_size_tuple), np.max(image_size_tuple))
                resized_masks = self.multidim_mask_func(mask_data, data_dims, self.resize_data_2D, adjusted_size, order)
                # If the original mask was padded and then resized to smaller dimensions, unpad it
                return self.multidim_mask_func(resized_masks, data_dims, self.unpad_to_original_size, image_size_tuple)
            mask_data = self.multidim_mask_func(mask_data, data_dims, self.pad_to_square)
            return self.multidim_mask_func(mask_data, data_dims, self.resize_data_2D, image_size_tuple, order)
        if self.make_square and (original_shape[0] > image_size_tuple[0] or original_shape[1] > image_size_tuple[1]):
            # Square mask larger than the target: resize to a square, then unpad
            side = image_size_tuple[1] if original_shape[0] > image_size_tuple[0] else image_size_tuple[0]
            resized_masks = self.multidim_mask_func(mask_data, data_dims, self.resize_data_2D, (side, side), order, False)
            return self.multidim_mask_func(resized_masks, data_dims, self.unpad_to_original_size, image_size_tuple)
        return self.multidim_mask_func(mask_data, data_dims, self.resize_data_2D, image_size_tuple, order)

    def prep_mask(self, mask_data: np.ndarray) -> tuple[np.ndarray, list, list]:
        """
        Prepare mask data (2D [H x W] or 3D [Slice x H x W]); returns (prepped mask, kept slice indices, all slice
        indices).
        """
        data_dims = self.data_dims(mask_data)
        assert data_dims == 2 or data_dims == 3, "Mask data must be 2D, or 3D numpy array."

        mask_data = self.remove_labels(mask_data)
        mask_data = self.remove_small_objects(mask_data)
        mask_data = np.uint8(mask_data)

        original_z = np.arange(np.shape(mask_data)[0]) if data_dims == 3 else [0]
        if self.verbose:
            print('orig_mask shape:', np.shape(mask_data))

        if self.crop_non_zero_slices_flag and data_dims == 3:
            # Labelled slices, plus an optional random share of all slices
            z_select = []
            if self.z_sample > 0:
                all_slices = np.arange(np.shape(mask_data)[0])
                np.random.shuffle(all_slices)
                z_select = all_slices[0:int(len(all_slices) * self.z_sample)]
            z_indices = np.unique(np.sort(list(find_non_zero_slices(mask_data)) + list(z_select)))
            if self.verbose:
                print('z_sample:', z_select)
                print('z_indices', z_indices)
            if len(z_indices) == 0:
                warnings.warn("Mask does not have labels", UserWarning)
            else:
                mask_data = crop_data_to_slices(mask_data, z_indices)
        else:
            z_indices = original_z

        preprocessed_mask = np.uint8(self.resize_mask(mask_data, order=0))
        if data_dims == 2:
            preprocessed_mask = np.expand_dims(preprocessed_mask, axis=0)
        if self.verbose:
            print(f"preprocessed_mask shape: {preprocessed_mask.shape}, dtype: {preprocessed_mask.dtype}, "
                  f"z_indices length: {len(z_indices)}")
        if 'dtype' in self.data_transforms:
            preprocessed_mask = transform_data(preprocessed_mask, {'dtype': self.data_transforms['dtype']})
        return (preprocessed_mask, z_indices, original_z)

    def get_prepped_data(self, filepath: str, no_dicom_ext_flag: bool = False) -> tuple[np.ndarray, list, list]:
        """Load and prepare one mask volume (``prep_mask``)."""
        mask_data, _ = self.get_data(filepath, no_dicom_ext_flag=no_dicom_ext_flag)
        return self.prep_mask(mask_data)

    def validate_mask(self, data: np.ndarray, expected_properties: dict):
        """Validate the data against expected properties."""
        if expected_properties:
            validate_data(data, expected_properties, content='mask')


def get_relabeled_bounding_boxes(multiclass_mask, relabel_dict: dict):

    # obtain unique labels in the image
    unique_labels = np.unique(multiclass_mask)
    bounding_boxes = []

    for label in unique_labels:
        # Skip background label
        if label == 0:
            continue
        
        # Check relabeling dictionary
        bb_label = label
        if relabel_dict:
            # Map label to a new bounding box label
            # The while loop allows for multiple maps (e.g. 1 --> 2 --> 3)
            # while int(bb_label) in relabel_dict:
            #     bb_label = relabel_dict[int(bb_label)]
            
            # Check dict for a single map, each label map is explicitly defined
            if int(bb_label) in relabel_dict.keys():
                bb_label = relabel_dict[int(bb_label)]
        
        # Original logic for bounding box of entire label
        y_indices, x_indices = np.where(multiclass_mask == label)

        # Calculate bounding box coordinates
        center_x, center_y, width, height = get_box_coords(x_indices, y_indices)

        # Normalize
        height_sz, width_sz = multiclass_mask.shape
        bounding_boxes.append(
            str(int(bb_label)) +
            ' ' + str(center_x / width_sz) +
            ' ' + str(center_y / height_sz) +
            ' ' + str(width / width_sz) +
            ' ' + str(height / height_sz) )

    return bounding_boxes

def get_box_coords(x_indices, y_indices):

    # Compute the bounding box for the selected component
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Calculate center and dimensions
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return center_x, center_y, width, height

def write_list_to_file(file_path, data_list, write_mode='w'):
    with open(file_path, write_mode) as file:
        for item in data_list:
            file.write(str(item) + '\n')