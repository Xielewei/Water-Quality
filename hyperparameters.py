import scipy.io
import numpy as np
import json
import os
import torch
from config import load_config
from trainer import Trainer
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

def save_history(history, log_dir, run_id, param_name, param_value):
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
        else:
            print(f"Warning: Object of type {type(obj)} is not JSON serializable. Converted to str.")
            return str(obj)

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{log_dir}/history_{param_name}{param_value}_run{run_id}_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(to_serializable(history), f, indent=4, ensure_ascii=False)
    print(f"✅ Training history saved to {filepath}")
    return filepath

def compute_metrics(histories):
    """Compute mean and standard deviation of test metrics over multiple runs."""
    test_mae = [float(h['test_mae']) for h in histories]
    test_mape = [float(h['test_mape']) for h in histories]
    test_rmse = [float(h['test_rmse']) for h in histories]
    
    metrics = {
        'test_mae_mean': float(np.mean(test_mae)),
        'test_mae_std': float(np.std(test_mae)),
        'test_mape_mean': float(np.mean(test_mape)),
        'test_mape_std': float(np.std(test_mape)),
        'test_rmse_mean': float(np.mean(test_rmse)),
        'test_rmse_std': float(np.std(test_rmse))
    }
    return metrics

def main():
    # Define hyperparameter values
    h_values = [6, 12, 24, 48]
    u_values = [1, 3, 5, 7]
    N_sub_values = [10, 15, 20, 25]
    unobs_N_sub_pairs = [(0.2, 25), (0.4, 18), (0.6, 12), (0.8, 6)]  # unobserved_proportion and N_sub pairs
    
    num_runs = 3  # Number of runs per hyperparameter combination
    config_file = "config_AUX_DGCN.yaml"  # Use AUX_DGCN config file
    log_dir = "parameter_logs"  # Shared log directory
    all_metrics = {}  # Store metrics for all hyperparameter combinations
    
    # Load base configuration
    base_cfg = load_config(config_file)
    
    # 1. Vary h, keep other parameters at default values
    for h in h_values:
        param_name = "h"
        param_value = h
        param_str = f"{param_name}{param_value}"
        print(f"\n=== Training parameter: {param_str} ===")
        
        histories = []
        for run_id in range(1, num_runs + 1):
            print(f"\nRun {run_id}/{num_runs} with {param_str}")
            
            # Copy base config and update h and related dimensions
            cfg = base_cfg.copy()
            cfg.data.h = h
            cfg.model.params.input_dim = h  # Update input_dim accordingly
            cfg.model.params.output_dim = h  # Update output_dim accordingly
            cfg.paths.log_dir = log_dir
            cfg.paths.save_path = f"{log_dir}/best_model_{param_str}_run{run_id}.pth"
            
            # Initialize trainer
            trainer = Trainer(cfg)
            
            # Train and obtain history
            history = trainer.fit(Y_train, Y_test, X_aux_train, X_aux_test, A)
            
            # Save history
            save_history(history, log_dir, run_id, param_name, param_value)
            
            histories.append(history)
        
        # Compute and store metrics
        metrics = compute_metrics(histories)
        all_metrics[param_str] = metrics
        
        # Print metrics
        print(f"\nMetrics for {param_str}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.6f}")
    
    # 2. Vary u, keep other parameters at default values
    for u in u_values:
        param_name = "u"
        param_value = u
        param_str = f"{param_name}{param_value}"
        print(f"\n=== Training parameter: {param_str} ===")
        
        histories = []
        for run_id in range(1, num_runs + 1):
            print(f"\nRun {run_id}/{num_runs} with {param_str}")
            
            # Copy base config and update u
            cfg = base_cfg.copy()
            cfg.data.u = u
            cfg.paths.log_dir = log_dir
            cfg.paths.save_path = f"{log_dir}/best_model_{param_str}_run{run_id}.pth"
            
            # Initialize trainer
            trainer = Trainer(cfg)
            
            # Train and obtain history
            history = trainer.fit(Y_train, Y_test, X_aux_train, X_aux_test, A)
            
            # Save history
            save_history(history, log_dir, run_id, param_name, param_value)
            
            histories.append(history)
        
        # Compute and store metrics
        metrics = compute_metrics(histories)
        all_metrics[param_str] = metrics
        
        # Print metrics
        print(f"\nMetrics for {param_str}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.6f}")
    
    # 3. Vary N_sub, keep other parameters at default values
    for N_sub in N_sub_values:
        param_name = "N_sub"
        param_value = N_sub
        param_str = f"{param_name}{param_value}"
        print(f"\n=== Training parameter: {param_str} ===")
        
        histories = []
        for run_id in range(1, num_runs + 1):
            print(f"\nRun {run_id}/{num_runs} with {param_str}")
            
            # Copy base config and update N_sub
            cfg = base_cfg.copy()
            cfg.data.N_sub = N_sub
            cfg.model.params.num_nodes = N_sub
            cfg.paths.log_dir = log_dir
            cfg.paths.save_path = f"{log_dir}/best_model_{param_str}_run{run_id}.pth"
            
            # Initialize trainer
            trainer = Trainer(cfg)
            
            # Train and obtain history
            history = trainer.fit(Y_train, Y_test, X_aux_train, X_aux_test, A)
            
            # Save history
            save_history(history, log_dir, run_id, param_name, param_value)
            
            histories.append(history)
        
        # Compute and store metrics
        metrics = compute_metrics(histories)
        all_metrics[param_str] = metrics
        
        # Print metrics
        print(f"\nMetrics for {param_str}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.6f}")
    
    # 4. Vary paired unobserved_proportion and N_sub
    for unobs, N_sub in unobs_N_sub_pairs:
        param_name = "unobs_Nsub"
        param_value = f"{unobs}_Nsub{N_sub}"
        param_str = f"{param_name}{unobs}_Nsub{N_sub}"
        print(f"\n=== Training parameter: {param_str} ===")
        
        histories = []
        for run_id in range(1, num_runs + 1):
            print(f"\nRun {run_id}/{num_runs} with {param_str}")
            
            # Copy base config and update unobserved_proportion and N_sub
            cfg = base_cfg.copy()
            cfg.data.unobserved_proportion = unobs
            cfg.data.N_sub = N_sub
            cfg.model.params.num_nodes = N_sub
            cfg.paths.log_dir = log_dir
            cfg.paths.save_path = f"{log_dir}/best_model_{param_str}_run{run_id}.pth"
            
            # Initialize trainer
            trainer = Trainer(cfg)
            
            # Train and obtain history
            history = trainer.fit(Y_train, Y_test, X_aux_train, X_aux_test, A)
            
            # Save history
            save_history(history, log_dir, run_id, param_name, param_value)
            
            histories.append(history)
        
        # Compute and store metrics
        metrics = compute_metrics(histories)
        all_metrics[param_str] = metrics
        
        # Print metrics
        print(f"\nMetrics for {param_str}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.6f}")
    
    # Save metrics for all parameter combinations to a summary file
    summary_filepath = f"{log_dir}/summary_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(log_dir, exist_ok=True)
    with open(summary_filepath, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Metrics for all parameter combinations have been saved to {summary_filepath}")

if __name__ == "__main__":
    main()
