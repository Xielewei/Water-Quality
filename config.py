# config.py
from omegaconf import OmegaConf
import torch

def load_config(config_path: str):
    cfg = OmegaConf.load(config_path)
    return cfg