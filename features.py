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

# Define auxiliary variable names
aux_variable_names = [
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius (Maximum)',
    'pH, water, unfiltered, field, standard units (Maximum)',
    'pH, water, unfiltered, field, standard units (Minimum)',
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius (Minimum)',
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius (Mean)',
    'Dissolved oxygen, water, unfiltered, milligrams per liter (Maximum)',
    'Dissolved oxygen, water, unfiltered, milligrams per liter (Mean)',
    'Dissolved oxygen, water, unfiltered, milligrams per liter (Minimum)',
    'Temperature, water, degrees Celsius (Mean)',
    'Temperature, water, degrees Celsius (Minimum)',
    'Temperature, water, degrees Celsius (Maximum)'
]

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
    config_file = "config_AUX_DGCN.yaml"
    model_name = "AUX_DGCN"
    num_runs = 3  # Number of runs per auxiliary variable
    log_dir = "logs_feature"  # New log directory
    
    # Store metrics for all auxiliary variables
    all_aux_metrics = {}
    
    for aux_idx, aux_name in enumerate(aux_variable_names):
        print(f"\n=== Testing auxiliary variable: {aux_name} (index {aux_idx}) ===")
        
        # Select the current auxiliary variable
        X_aux_train_single = X_aux_train[:, :, aux_idx:aux_idx+1]  # (37, 423, 1)
        X_aux_test_single = X_aux_test[:, :, aux_idx:aux_idx+1]    # (37, 282, 1)
        
        histories = []
        for run_id in range(1, num_runs + 1):
            print(f"\nRun {run_id}/{num_runs} of {model_name} with auxiliary variable {aux_name}")
            
            # Load configuration
            cfg = load_config(config_file)
            
            # Set aux_dim to 1
            cfg.model.params.aux_dim = 1
            
            # Update save path and log directory to reflect the auxiliary variable
            cfg.paths.log_dir = f"{log_dir}/{aux_name.replace(', ', '_').replace(' ', '_')}"
            cfg.paths.save_path = f"./checkpoints/best_model_aux_dgcn_{aux_name.replace(', ', '_').replace(' ', '_')}_run_{run_id}.pth"
            
            # Initialize trainer
            trainer = Trainer(cfg)
            
            # Train and obtain history
            history = trainer.fit(Y_train, Y_test, X_aux_train_single, X_aux_test_single, A)
            
            # Add auxiliary variable information to history
            history['aux_variable'] = aux_name
            history['aux_index'] = aux_idx
            
            # Save history
            save_history(history, cfg.paths.log_dir, run_id)
            
            histories.append(history)
        
        # Compute and store metrics for this auxiliary variable
        metrics = compute_metrics(histories)
        metrics['aux_variable'] = aux_name
        metrics['aux_index'] = aux_idx
        all_aux_metrics[f"aux_{aux_idx}_{aux_name.replace(', ', '_').replace(' ', '_')}"] = metrics
        
        # Print metrics for this auxiliary variable
        print(f"\nMetrics for {aux_name}:")
        for metric_name, value in metrics.items():
            if metric_name not in ['aux_variable', 'aux_index']:
                print(f"{metric_name}: {value:.6f}")
    
    # Save metrics of all auxiliary variables to a summary file
    summary_filepath = f"{log_dir}/summary_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(log_dir, exist_ok=True)
    with open(summary_filepath, "w", encoding="utf-8") as f:
        json.dump(all_aux_metrics, f, indent=4, ensure_ascii=False)
    print(f"\n✅ All auxiliary variable metrics have been saved to {summary_filepath}")

if __name__ == "__main__":
    main()
