# utils.py
from models import MODEL_REGISTRY
import torch
import torch.nn as nn
from typing import Callable

def build_model(cfg_model):
    """
    Build a model from the config.
    """
    name = cfg_model.name
    params = cfg_model.params

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry. Available: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[name](**params)
    
    return model

# Supported loss function mapping
LOSS_MAP = {
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,           # MAE
    "SmoothL1Loss": nn.SmoothL1Loss, # Huber
    "HuberLoss": nn.HuberLoss,     # PyTorch 1.10+
}

def build_loss(cfg) -> Callable:
    """
    Build a loss function from the configuration.
    Args:
        cfg: OmegaConf config object containing loss.name and loss.params
    Returns:
        Loss function instance
    """
    loss_name = cfg.loss.name
    loss_params = cfg.loss.params or {}

    if loss_name not in LOSS_MAP:
        raise ValueError(f"Unsupported loss function: {loss_name}, choose from {list(LOSS_MAP.keys())}")

    loss_class = LOSS_MAP[loss_name]
    return loss_class(**loss_params)
