import os
from string import Template

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# --------------- Main class to load and manipulate config files ---------------
from src.utils.file_management.config_loader import ConfigPathManager
from src.utils.file_management.args_handler import apply_overrides


def get_config_name(cli_value=None):
    """
    Resolve config name from env or CLI for consistent SLURM/env and CLI usage.
    Returns CONFIG_NAME from environment if set and non-empty; otherwise cli_value.
    Strips a trailing .yaml from the chosen value so load_main_config / extract_config_path
    can add it consistently.
    """
    name = os.environ.get("CONFIG_NAME", "").strip() or (cli_value if cli_value is not None else None)
    if name is None:
        return None
    return name[:-5] if name.lower().endswith(".yaml") else name

# -------- Tool blocks to load and combine config files for experimental use cases --------


    

# ------------ Primary Pipeline Functions to load and manipulate config files ------------

# ------------------------------------------------------------------


# ------------------------------------------------------------------
def load_main_config(config_file_name, base_dir, sub_dir=None, **kwargs):
    """Load the main dataset/experiment YAML. base_dir is the project root; the function resolves the config file under config/<sub_dir>/ (e.g. config/datasets/). Returns merged config dict with env substitution and optional overrides applied."""
    configPathManager = ConfigPathManager(base_dir=base_dir)

    # Extract path to config and make sure yaml file exists
    config_path = configPathManager.extract_config_path(file_name=config_file_name, 
                                                        sub_dir=sub_dir)
    main_config = configPathManager.load_config_yaml_path(config_path)

    # Process the configuration to replace variables
    for key, value in main_config.items():
        if isinstance(value, str) and '$' in value:
            # Create a Template and substitute variables
            template = Template(value)
            main_config[key] = template.safe_substitute(os.environ, **main_config)
    
    # Apply terminal argument updates
    if kwargs:
        apply_overrides(main_config, kwargs)

    return main_config 


    
# ---------------- SUMMARIZE CONFIG FILE ------------------ #
def summarize_config(config, path):
    def _summarize(current_config, indent_level=0):
        summary_lines = []
        indent_space = '  ' * indent_level  # Two spaces per indent level

        for key, value in current_config.items():
            if isinstance(value, dict):
                summary_lines.append(f"{indent_space}{key}:")
                summary_lines.extend(_summarize(value, indent_level + 1))
            else:
                summary_lines.append(f"{indent_space}{key}: {value}")

        return summary_lines
    
    # Check if the path exists, if not, create it
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # This will create all intermediate-level directories needed to contain the leaf directory

    summary_lines = _summarize(config)
    summary_text = '\n'.join(summary_lines)

    with open(f"{path}/config_file_summary.txt", "w") as summary_file:
        summary_file.write(summary_text)

# ------------- Dynamic Dataset Cfg Updating ------------- #
