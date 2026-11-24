import os
import yaml
import warnings
warnings.filterwarnings('ignore')
# -------------------- Basic Tools --------------------

def load_yaml(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file) or {}
    return config

def flatten_context(context, parent_key='', sep='.'):
    items = []
    for k, v in context.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_context(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def process_config_values(config):

    # Flatten the configuration context for dot notation support
    flat_config = flatten_context(config)
    
    # Function to substitute placeholders with actual values
    def substitute(current_config):
        if isinstance(current_config, dict):
            for key, value in current_config.items():
                current_config[key] = substitute(value)
        elif isinstance(current_config, list):
            return [substitute(element) for element in current_config]
        elif isinstance(current_config, str):
            # Detect placeholder and substitute with the value from flat_config
            while '${' in current_config:
                start_index = current_config.find('${')
                end_index = current_config.find('}', start_index)
                if start_index == -1 or end_index == -1:
                    break  # No substitution needed
                placeholder = current_config[start_index + 2:end_index]
                if placeholder in flat_config:
                    # Convert the value to a string if it's not already a string
                    substitute_value = str(flat_config[placeholder])
                    # Replace placeholder with actual value
                    current_config = current_config.replace(
                        f'${{{placeholder}}}', substitute_value)
                else:
                    break  # Placeholder not found in flat_config
        return current_config
    
    # Substitute placeholders in the original nested config
    return substitute(config)


# --------------- Main Class to load and manipulate config files ---------------
    
class ConfigPathManager():
    """
    Class to manage configs paths and extract info. 
    """
    def __init__(self, base_dir:str):
        """
        Params:
        - base_dir: directory containing all 'config' files for project.
        """
        self.base_dir = os.path.join(base_dir, 'config')
        self.config_cache = {}  # Cache for loaded configurations

    def extract_config_path(self, file_name:str, sub_dir:str=None):
        """Return the full path to the config file. base_dir (passed at init) is the project root; the class joins config/ internally, so the path is base_dir/config/<sub_dir>/<file_name>."""

        if not file_name.lower().endswith('.yaml'):
            file_name += '.yaml'

        if sub_dir is not None:
            config_path = os.path.join(self.base_dir, 
                                    sub_dir, 
                                    file_name)
        else:
            config_path = os.path.join(self.base_dir, 
                                    file_name)
        
        if not os.path.exists(config_path):
            raise ValueError(f'Invalid full path to configuration file: {config_path}')
        return config_path
    
    # Each config file is read once per process.
    def load_config_yaml_path(self, config_path:str):
        """Load yaml and preprocess contents with caching."""
        if config_path not in self.config_cache:
            config = load_yaml(config_path)
            processed_config = process_config_values(config)
            self.config_cache[config_path] = processed_config
        return self.config_cache[config_path]


    

