# main.py
from config import load_config
from trainer import Trainer
import scipy.io
import numpy as np
import json
import os
import torch

# ==== Load the .mat file ====
mat = scipy.io.loadmat('datasets/water_dataset.mat')
Y_train = mat["Y_tr"]
Y_test = mat["Y_te"]
X_aux_tr = mat["X_tr"]
X_aux_te = mat["X_te"]
A = np.load("datasets/adjacency_matrix.npy")

X_aux_train_list = [X_aux_tr[0, i] for i in range(X_aux_tr.shape[1])]
X_aux_train_stacked = np.stack(X_aux_train_list, axis=0)  # -> (423, 37, 11)
X_aux_train = X_aux_train_stacked.transpose(1, 0, 2)      # -> (37, 423, 11)

# === Process test set X_test ===
X_aux_test_list = [X_aux_te[0, i] for i in range(X_aux_te.shape[1])]
X_aux_test_stacked = np.stack(X_aux_test_list, axis=0)    # -> (282, 37, 11)
X_aux_test = X_aux_test_stacked.transpose(1, 0, 2)        # -> (37, 282, 11)


def save_history(history, log_dir):
    """Safely save the training history as JSON, compatible with NumPy data types"""
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # For non-serializable types, convert to string (optional: can skip)
            print(f"Warning: Object of type {type(obj)} is not JSON serializable. Converted to str.")
            return str(obj)

    os.makedirs(log_dir, exist_ok=True)
    filepath = f"{log_dir}/history.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(to_serializable(history), f, indent=4, ensure_ascii=False)
    print(f"✅ Training history saved to {filepath}")

def main():
    # Load configuration
    cfg = load_config("config_AUX_DGCN.yaml")

    # Initialize trainer
    trainer = Trainer(cfg)

    # Start training
    history = trainer.fit(Y_train, Y_test, X_aux_train, X_aux_test, A)

    # Save training history (safe serialization)
    save_history(history, cfg.paths.log_dir)
    
if __name__ == "__main__":
    main()
