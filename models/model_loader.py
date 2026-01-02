"""
Model loader for loading models from configuration.
Author: chandu
"""

from typing import Dict
import yaml
from models.model_wrapper import EmbeddingModel, LocalModel


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict) -> None:
    """
    Validate configuration file.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    if 'models' not in config:
        raise ValueError("Configuration missing 'models' section")

    if not config['models']:
        raise ValueError("No models defined in configuration")

    for model in config['models']:
        if 'id' not in model:
            raise ValueError("Model missing 'id' field")
        if 'type' not in model:
            raise ValueError(f"Model {model.get('id')} missing 'type' field")
        if model['type'] not in ['local']:
            raise ValueError(f"Model {model.get('id')} has invalid type: {model['type']}")
        if 'model_name' not in model:
            raise ValueError(f"Model {model.get('id')} missing 'model_name' field")


def load_models_from_config(config_path: str) -> Dict[str, EmbeddingModel]:
    """
    Load all models specified in configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary mapping model IDs to model instances
    """
    config = load_config(config_path)
    validate_config(config)

    models = {}

    for model_config in config['models']:
        model_id = model_config['id']
        model_type = model_config['type']

        try:
            if model_type == 'local':
                model = LocalModel(
                    model_name=model_config['model_name'],
                    model_id=model_id,
                    dimensions=model_config.get('dimensions', 384)
                )
                models[model_id] = model
            else:
                print(f"Warning: Unknown model type '{model_type}' for model {model_id}, skipping...")

        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            print(f"Skipping model {model_id}...")
            continue

    if not models:
        raise ValueError("Failed to load any models")

    return models
