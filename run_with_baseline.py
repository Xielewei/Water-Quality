import scipy.io
import numpy as np
import json
import os
import torch
from config import load_config
from trainer import Trainer
import glob
from datetime import datetime

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

X_aux_test_list = [X_aux_te[0, i] for i in range(X_aux_te.shape[1])]
X_aux_test_stacked = np.stack(X_aux_test_list, axis=0)    # -> (282, 37, 11)
X_aux_test = X_aux_test_stacked.transpose(1, 0, 2)        # -> (37, 282, 11)

def save_history(history, log_dir, run_id):
    """Safely save history as JSON, compatible with NumPy data types."""
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # For types that cannot be serialized, convert to string as a last resort
            print(f"Warning: Object of type {type(obj)} is not JSON serializable. Converted to str.")
            return str(obj)

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{log_dir}/history_run_{run_id}_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(to_serializable(history), f, indent=4, ensure_ascii=False)
    print(f"✅ Training history saved to {filepath}")
    return filepath

def compute_metrics(histories):
    """Compute mean and standard deviation of test metrics over multiple runs."""
    test_mae = [float(h['test_mae']) for h in histories]  # Convert to Python float
    test_mape = [float(h['test_mape']) for h in histories]
    test_rmse = [float(h['test_rmse']) for h in histories]
    
    metrics = {
        'test_mae_mean': float(np.mean(test_mae)),  # Ensure returning a Python float
        'test_mae_std': float(np.std(test_mae)),
        'test_mape_mean': float(np.mean(test_mape)),
        'test_mape_std': float(np.std(test_mape)),
        'test_rmse_mean': float(np.mean(test_rmse)),
        'test_rmse_std': float(np.std(test_rmse))
    }
    return metrics

def main():
    # Get all YAML configuration files
    config_files = glob.glob("config_*.yaml")
    num_runs = 5  # Number of runs per model
    
    # Store metrics for all models
    all_model_metrics = {}
    
    for config_file in config_files:
        model_name = config_file.replace(".yaml", "").replace("config_", "")
        print(f"\n=== Training model: {model_name} ===")
        
        histories = []
        for run_id in range(1, num_runs + 1):
            print(f"\nRun {run_id}/{num_runs} of {model_name}")
            
            # Load configuration
            cfg = load_config(config_file)
            
            # Initialize trainer
            trainer = Trainer(cfg)
            
            # Train and obtain history
            history = trainer.fit(Y_train, Y_test, X_aux_train, X_aux_test, A)
            
            # Save history with run index and timestamp
            save_history(history, cfg.paths.log_dir, run_id)
            
            histories.append(history)
        
        # Compute and store metrics for this model
        metrics = compute_metrics(histories)
        all_model_metrics[model_name] = metrics
        
        # Print metrics for this model
        print(f"\nMetrics for {model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.6f}")
    
    # Save metrics for all models to a summary file
    summary_filepath = f"logs/summary_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("logs", exist_ok=True)
    with open(summary_filepath, "w", encoding="utf-8") as f:
        json.dump(all_model_metrics, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Metrics for all models have been saved to {summary_filepath}")

if __name__ == "__main__":
    main()
