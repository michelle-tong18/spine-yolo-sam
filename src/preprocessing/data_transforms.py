import numpy as np
import cc3d


def transform_data(data, transform_dict:dict={}):
    """
    Apply a series of transformations to the input data based on the provided dictionary.
    Args:
        data: Input numpy array to be transformed.
        transform_dict: Dictionary where keys are transformation names and values are parameters.
    Returns:
        Transformed numpy array.
    """

    # Check whether to perform transforms
    if not transform_dict:
        return data

    assert len(np.shape(data))==2 or len(np.shape(data))==3, "Data must be 2D, or 3D numpy array."
    
    # Apply transforms sequentially
    for transform_name, transform_args in transform_dict.items():
        if transform_name == 'transpose':
            data = np.transpose(data, transform_args) # args are the axes

        elif transform_name == 'swapaxes':
            data = np.swapaxes(data, transform_args[0], transform_args[1]) #args are the axes 

        elif transform_name == 'flip':
            data = np.flip(data, axis=transform_args) # args is the axis

        elif transform_name == 'rot90':
            data = np.rot90(data, axes=transform_args) #args are the axes 

        elif transform_name == 'pred_to_binary':
            data[data <= 0.5] = 0
            data[data > 0.5] = 1

        elif transform_name == 'remove_values_below':
            data[data<transform_args] = 0

        elif transform_name == 'remove_values_above':
            data[data>transform_args] = 0
            
        elif transform_name == 'drop_dim': # remove axes with dim 1
            if transform_args[0]==0:
                data = np.squeeze(data[transform_args[1]])
            elif transform_args[0]==1:
                data = np.squeeze(data[:,transform_args[1]])
            elif transform_args[0]==2:
                data = np.squeeze(data[:,:,transform_args[1]])
            elif transform_args[0]==3:
                data = np.squeeze(data[:,:,:,transform_args[1]])
            else:
                raise ValueError('Invalid input for transform drop_dims')
                
        elif transform_name == 'top_cc_3D':
            data = top_largest_cc(data, transform_args, connectivity=26)
            
        elif transform_name == 'top_cc_2D':
            for slice_index in range(np.shape(data)[0]):
                data[slice_index] = top_largest_cc(data[slice_index], transform_args, connectivity=8)
            
        elif transform_name == 'remove_small_cc_3D':
            if transform_args > 0:
                data = remove_small_cc(data, connectivity=26, pixel_threshold=transform_args)

        elif transform_name == 'remove_small_cc_2D':
            if transform_args > 0:
                if len(np.shape(data)) == 2:
                    data = remove_small_cc(data, connectivity=8, pixel_threshold=transform_args)
                elif len(np.shape(data)) == 3:
                    for slice_index in range(np.shape(data)[0]):
                        data[slice_index] = remove_small_cc(data[slice_index], connectivity=8, pixel_threshold=transform_args)

        elif transform_name == 'border':
            # mask out border, assuming slice dim==0 for 3D
            if len(np.shape(data))==2:
                data[0:transform_args,:] = 0
                data[-transform_args:,:] = 0
                data[:,0:transform_args] = 0
                data[:,-transform_args:] = 0
            elif len(np.shape(data))==3:
                data[:, 0:transform_args,:] = 0
                data[:, -transform_args:,:] = 0
                data[:, :,0:transform_args] = 0
                data[:, :,-transform_args:] = 0

        elif transform_name == 'rezero':
            # re-zero data to start from 0
            if transform_args:
                min_pv = np.min(data)
                data -= np.min(data)

        elif transform_name == 'clip': 
            # clip data to min/max = transform_args[0], transform_args[1]
            data = np.clip(data, transform_args[0], transform_args[1])
            
        elif transform_name == 'percentile':
            # clip data to percentile value
            if isinstance(transform_args, int):
                percentile = np.clip([0, transform_args],0,100)
            elif len(transform_args)==2:
                percentile = np.clip([transform_args[0],transform_args[1]],0,100)

            lower_bound, upper_bound = np.percentile(data, percentile)
            if lower_bound == upper_bound:
                raise ValueError("Percentile bounds are equal; cannot clip data.")
            else:
                data = np.clip(data, lower_bound, upper_bound)
            
        elif transform_name == 'normalize':
            # normalize data to range 0 - max of data type or max of input data
            if isinstance(transform_args, int):
                max_pv = float(transform_args)
            elif isinstance(transform_args, float):
                max_pv = float(transform_args)
            elif transform_args == '1':
                max_pv = 1
            elif transform_args == '255':
                max_pv  = 255
            elif transform_args == 'np.uint8':
                max_pv  = 2**8 - 1
            elif transform_args == 'np.uint16':
                max_pv  = 2**16 - 1
            elif transform_args == 'np.uint32':
                max_pv  = 2**32 - 1
            elif transform_args == 'np.uint64':
                max_pv  = 2**64 - 1
            elif transform_args == 'data_max':
                max_pv  = np.max(data)
            else:
                max_pv  = np.max(data)
                
            data = data.astype(np.float64)
            data -= np.min(data)
            if np.max(data)>0:
                data /= np.max(data)
            else:
                data = np.abs(data)
                data /= np.max(data)
                
            data *= max_pv
            
        elif transform_name == 'dtype':
            
            if 'int' in transform_args:
                data = np.round(data)
            n_vals = len(np.unique(data))
            
            if transform_args == 'np.uint8':
                data = data.astype(np.uint8)
            elif transform_args == 'np.uint16':
                data = data.astype(np.uint16)
            elif transform_args == 'np.uint32':
                data = data.astype(np.uint32)
            elif transform_args == 'np.uint64':
                data = data.astype(np.uint64)
            elif transform_args == 'np.int8':
                data = data.astype(np.int8)
            elif transform_args == 'np.int16':
                data = data.astype(np.int16)
            elif transform_args == 'np.int32':
                data = data.astype(np.int32)
            elif transform_args == 'np.int64':
                data = data.astype(np.int64)
            elif transform_args == 'np.float32':
                data = data.astype(np.float32)
            elif transform_args == 'np.float64':
                data = data.astype(np.float64)
            else:
                raise ValueError('Invalid input for transform dtype')

            if n_vals != len(np.unique(data)):
                print(data.dtype, transform_args, n_vals, len(np.unique(data)))
                raise ValueError("Information lost during data conversion")
            
        elif transform_name == 'contrast':
            # increase contrast by transform_args [alpha,beta] factor around mean
            data = data.astype(np.float64)
            data_mean = np.mean(data)

            # zero the data around the mean
            data -= data_mean

            # adjust contrast with alpha and beta
            if transform_args is None:
                alpha = 1.25
                beta = 100
            else:
                alpha = transform_args[0]
                beta = transform_args[1]
                
            data = alpha*data + beta

            # re-zero data
            data += data_mean

        elif transform_name == 'collapse_label_dim':
            data_ref = np.copy(data[:,:,:,transform_args[0]])
            label_counter_list = transform_args[1] 
            for index in range(np.shape(data_ref)[3]):
                label_counter = label_counter_list[index]
                if index == 0:
                    data = np.copy(data_ref[:,:,:,index]*label_counter)
                else:
                    # Conditional addition to avoid overwriting existing labels
                    data = np.where(data == 0, data_ref[:,:,:,index]*label_counter, data)
        
        elif transform_name == 'remap_values':
            relabel_dict = transform_args.copy()
            data_old = data.copy()
            data = np.zeros_like(data_old)
            for label in np.unique(data_old):
                if int(label) in relabel_dict.keys():
                    data[data_old==label] = relabel_dict[int(label)]
                else:
                    data[data_old==label] = label

    return data

def top_largest_cc(data, k_largest:int, connectivity:int):
    '''
    Label by connected components and keep k_largest largest non-background labels.
    '''
    # Get a labeling of the k largest objects in the image.
    # The output will be relabeled from 1 to N.
    largest_k_data, N = cc3d.largest_k(data, k=k_largest, connectivity=connectivity, delta=0, return_N=True)
    # Use original labels and update labels to zero if not part of k largest objects
    data = data * (largest_k_data > 0) 
    return data

def remove_small_cc(data, connectivity:int, pixel_threshold:int):
    '''
    Remove small connected components in 3D or 2D data based on pixel thresholds.
    '''
    data = cc3d.dust(data, threshold=pixel_threshold, connectivity=connectivity, in_place=True)
    return data

def find_non_zero_slices(mask_data: np.ndarray) -> np.ndarray:
    """Identify indices of slices that contain non-zero values."""
    z_indices = np.unique(np.where(mask_data > 0)[0])
    return z_indices

def extrapolate_z(z_indices, shape_in, shape_out):
    
    z_in = shape_in - 1
    z_out = shape_out - 1
    
    z_indices = np.clip(z_indices,0,z_in)
    z_indices = z_out*(z_indices/z_in)
    z_indices = np.round(np.clip(z_indices,0,z_out)).astype(int)
    
    return z_indices

def crop_data_to_slices(data: np.ndarray, z_indices: np.ndarray) -> np.ndarray:
    """
    Crop the 3D data to only include slices with non-zero values. Assumes slice dimension is 0.
    Args:
        data: 3D numpy array [Slice x H x W]
        z_indices: Indices of slices to retain
    Returns:
        Cropped 3D numpy array.
    """
    return data[z_indices]

def data_properties(data, content='mask'):
    """
    Extracts properties about the data into a dictionary.
    """
    assert content in ['image', 'mask'], ValueError('Invalid option for data content variable')
    
    def number_connected_components(data, connectivity=26):
        labeled_data, num_labels = cc3d.connected_components(data, connectivity=connectivity, return_N=True)
        return num_labels
    
    data_info = {
        'shape': np.shape(data),
        'max': np.max(data),
        'min': np.min(data),
        'smallest_dim': np.argmin(np.shape(data)) if data is not None else None,
        'largest_dim': np.argmax(np.shape(data)) if data is not None else None,
        'number_unique_values': len(np.unique(data)),
        'type': data.dtype,
        'finite_values': np.all(np.isfinite(data)), #TODO # Replace NaNs and infs with zero or a specified value --> valid_vol =  np.nan_to_num(vol) 
        'masked_array': np.ma.is_masked(data), #TODO # Replace masked values with zero --> valid_vol = valid_vol.filled(0) 
    }

    if content == 'mask':
        data_info['unique_values'] = sorted(np.unique(data))
        #data_info['number_connected_components'] = number_connected_components(data) # longer runtime

    return data_info

def validate_data_properties(actual_properties:dict, expected_properties:dict):
    """
    Checks whether actual and expected data properties match.
    
    Params:
        actual_properties: Properties of data. Keys must match expected_properties.
        expected_properties: Properties we expect the data to have. Keys must match actual_properties.
    Returns:
        matching_properties(bool): True/False
    """
    matching_properties = True
    for property_name, expected_val in expected_properties.items():
        actual_val = actual_properties[property_name]
        #print(property_name, actual_val)
        if actual_val != expected_val:
            matching_properties = False
            #raise UserWarning(f'Unexpected data {property_name}\n\tExpected {expected_val}\n\tActual {actual_val}.')
            raise ValueError(f'Unexpected data {property_name}\n\tExpected {expected_val}\n\tActual {actual_val}.')
    return matching_properties

def validate_data(data, expected_properties:dict, content='mask'):
    """
    Determines whether data is consistent with what is expected.
    """
    actual_properties = data_properties(data, content)
    return validate_data_properties(actual_properties, expected_properties)