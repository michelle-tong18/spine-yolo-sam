import os
import pandas as pd
from typing import Union


def extract_data_pair_paths(data_dir:str, img_column:Union[list[str], str, None] = None, mask_column:Union[str, None] = None, include_accession=False):
    """
    Reads data_dir to determine paths to data and the associated subject ids.
    Args:
        data_dir: csv of the original data with paired image and mask paths.
        columns_to_process: Relevant csv columns names. List of strings or string.
    Returns:
        data_paths_info: tuple of tuples 
                        for a csv ((path to the image, path to the mask, subject id))
    """
    
    # Check whether data directory is valid
    assert os.path.exists(data_dir) and data_dir.endswith('.csv'), \
            "Error: The path provided in 'data_dir' is not a CSV file."
    
    # Create a list of tuples that specifies the path(s) to data and subject id 
    data_paths_info = []
    
    # Load files for subjects
    if data_dir.endswith('.csv'):
        subjects_df = pd.read_csv(data_dir)
        
        # Iterate over each row in the DataFrame, processing based on the configuration
        for _, row in subjects_df.iterrows():

            # Ensure image column names are a list, converts single string to list
            if not isinstance(img_column, list):
                img_column = [img_column]
            # Dynamically determine columns to process based on presence in the row
            img_paths = [row[column].strip() for column in img_column \
                                if pd.notnull(row[column])]
            # Check for valid input
            if not isinstance(img_paths, list):
                raise ValueError('Invalid image column name. Does not match .csv file.')
            
            # Mask paths optional (image-only inference / standardization)
            mask_paths = []
            if mask_column is not None and mask_column != "":
                # Ensure mask column names are a list, converts single string to list
                if not isinstance(mask_column, list):
                    mask_column = [mask_column]
                # Dynamically determine columns to process based on presence in the row
                mask_paths = [row[column].strip() for column in mask_column \
                                    if pd.notnull(row[column])]
                # Check for valid input
                if not isinstance(mask_paths, list):
                    raise ValueError('Invalid mask column name. Does not match .csv file.')
            
            # get subject id for filename
            subject_id = str(row['subject_id'])
            # check for accession
            if include_accession and 'accession' in subjects_df.columns:
                acc = row['accession']
                subject_id = subject_id + '_' + str(acc)

            data_paths_info.append((img_paths, mask_paths, subject_id))
    else:
        raise ValueError("The path provided is neither a valid directory nor a CSV file.")

    return data_paths_info
    


# ---------------- Npy Volume ID Path Functions ----------------

    


# ---------------- MISC ----------------
