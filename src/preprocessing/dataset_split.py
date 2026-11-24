"""
Split the preprocessed YOLO slices (slice_preprocessed/SAM/{png,labels}) into slice_preprocessed/YOLO/{train,val,test}
(symlinks) by the subjects of ``split_id`` in config/splits/splits_registry.yaml, and write the Ultralytics
``yolo_data.yaml``.

    python src/preprocessing/dataset_split.py <config>
"""
import os
import random
import numpy as np
import argparse
import yaml
from tqdm import tqdm

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.utils.file_management.config_handler import load_main_config
from src.sam_finetuning.datamodule.dynamic_dataset_loader import get_yolo_path, get_sam_path, get_splits

def split_dataset(config_name):
    """
    Parses config parameters to split standardized PNG images and text files for YOLO model training
    """

    cfg = load_main_config(config_name, base_dir=root, sub_dir='datasets')
    
    # Set the seeds for reproducibility
    seed = cfg.get('SEED', 42)
    random.seed(seed)
    np.random.seed(seed)
    
    # get splits from config
    splits =  get_splits(cfg)
    print('Data splits: ')
    for k,v in splits.items():
        print(k, v)
    
    # create yolo paths
    split_dst_dir = get_yolo_path(cfg)
    os.makedirs(os.path.join(split_dst_dir,'train','images'), exist_ok=True)
    os.makedirs(os.path.join(split_dst_dir,'val','images'), exist_ok=True)
    os.makedirs(os.path.join(split_dst_dir,'test','images'), exist_ok=True)
    os.makedirs(os.path.join(split_dst_dir,'train','labels'), exist_ok=True)
    os.makedirs(os.path.join(split_dst_dir,'val','labels'), exist_ok=True)
    os.makedirs(os.path.join(split_dst_dir,'test','labels'), exist_ok=True)

    split_accession = cfg.get('split_accession', False)
    # get standardized data
    split_src_dir = get_sam_path(cfg)
    if split_src_dir is not None:
        label_dir = os.path.join(split_src_dir,'labels')
        png_dir = os.path.join(split_src_dir,'png')
        if os.path.exists(label_dir):
            
            txt_files = os.listdir(label_dir)
            
            for f in tqdm(txt_files):
                src_file = os.path.join(label_dir, f)
                src_path, src_name = os.path.split(src_file)
                src_base, src_ext = os.path.splitext(src_name)
                f_parts = src_base.rsplit('-', 1)  # {volume}-{slice}
                if split_accession:
                    f_parts = f_parts[0].rsplit('_', 1)  # {subject}_{accession}
                pat_id = f_parts[0]
                
                if str(pat_id) in splits['val']:
                    data_split = 'val'
                elif str(pat_id) in splits['test']:
                    data_split = 'test'
                elif str(pat_id) in splits['train']:
                    data_split = 'train'
                else:
                    print('warning pat_id', pat_id,' not found')
                    data_split = 'train'

                dest_file = os.path.join(split_dst_dir,data_split,'labels',src_name)
                if os.path.exists(dest_file):
                    os.remove(dest_file)
                    os.symlink(src_file, dest_file)
                else:
                    os.symlink(src_file, dest_file)
                
                src_png  = os.path.join(split_src_dir,'png',src_base+'.png')
                dest_png = os.path.join(split_dst_dir,data_split,'images',src_base+'.png')
                # do not need to overwrite because using symlink
                if os.path.exists(src_png):
                    if os.path.exists(dest_png):
                        os.remove(dest_png)
                        os.symlink(src_png, dest_png)
                    else:
                        os.symlink(src_png, dest_png)
    
    # create data yaml file for Ultralytics YOLO
    yaml_file = gen_yolo_data_cfg(cfg)
    print('YOLO data yaml file: ', yaml_file)



def gen_yolo_data_cfg(cfg):
    '''
    Generate yaml file for Ultralytics YOLO data set
    '''
    
    # yolo paths
    yolo_path = get_yolo_path(cfg)
    
    # get mask labels
    mask_labels = cfg.get('mask_labels',{0: 'background'})
    
    # yolo data configuration
    yolo_config = {
        'path': yolo_path,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': mask_labels,
        'nc': len(mask_labels),
    }
    
    # save yolo config file
    yaml_file = os.path.join(yolo_path,'yolo_data.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(yolo_config, f, default_flow_style=False, sort_keys=False)
        
    return yaml_file
    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Preprocess medical data for SAM analysis.")
    parser.add_argument("config_name", help="Name of the YAML configuration file")
    args = parser.parse_args()

    # Load the configuration file
    split_dataset(args.config_name)