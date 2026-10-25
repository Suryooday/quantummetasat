"""Configuration loading and management utilities."""
import os
from typing import Any, Dict
import yaml

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.
    
    Args:
        file_path: Path to YAML configuration file.
        
    Returns:
        Dictionary with parsed configuration parameters.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}

def get_full_config(config_dir: str = "configs") -> Dict[str, Any]:
    """Load all standard configuration files merged into a single dictionary.
    
    Args:
        config_dir: Directory containing configuration YAML files.
        
    Returns:
        Consolidated configuration dictionary.
    """
    default_cfg = load_yaml(os.path.join(config_dir, "default.yaml"))
    dataset_cfg = load_yaml(os.path.join(config_dir, "dataset.yaml")) if os.path.exists(os.path.join(config_dir, "dataset.yaml")) else {}
    classical_cfg = load_yaml(os.path.join(config_dir, "classical.yaml")) if os.path.exists(os.path.join(config_dir, "classical.yaml")) else {}
    maml_cfg = load_yaml(os.path.join(config_dir, "maml.yaml")) if os.path.exists(os.path.join(config_dir, "maml.yaml")) else {}
    quantum_cfg = load_yaml(os.path.join(config_dir, "quantum.yaml")) if os.path.exists(os.path.join(config_dir, "quantum.yaml")) else {}
    
    merged = {**default_cfg}
    merged["dataset_config"] = dataset_cfg
    merged["classical_config"] = classical_cfg
    merged["maml_config"] = maml_cfg
    merged["quantum_config"] = quantum_cfg
    return merged
