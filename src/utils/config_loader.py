import yaml
import os

def load_config(config_path="config/config.yaml"):
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML config file.
        
    Returns:
        dict: Configuration dictionary.
    """
    # Adjust path if running from root or src
    if not os.path.exists(config_path):
        # Try finding it relative to the project root assuming script is run from project root
        if os.path.exists(os.path.join(os.getcwd(), config_path)):
            config_path = os.path.join(os.getcwd(), config_path)
        else:
             raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
