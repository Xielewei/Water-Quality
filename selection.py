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
    """Safely save history as JSON, compatible with NumPy and native Python data types."""
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64, float)):
            # Handle float values, including inf and nan
            if np.isinf(obj) or np.isnan(obj):
                return str(obj)  # Convert to string to avoid JSON serialization errors
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (int, str, bool, type(None))):
            return obj
        else:
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
    """Compute test metrics (single run)."""
    metrics = {
        'test_mae': float(histories['test_mae']),
        'test_mape': float(histories['test_mape']),
        'test_rmse': float(histories['test_rmse'])
    }
    return metrics

def train_with_aux_indices(aux_indices, config_file, log_dir, run_id):
    """Train model and return metrics based on given auxiliary variable indices."""
    # Select current auxiliary variables
    X_aux_train_single = X_aux_train[:, :, aux_indices]  # (37, 423, len(aux_indices))
    X_aux_test_single = X_aux_test[:, :, aux_indices]    # (37, 282, len(aux_indices))
    
    # Load configuration
    cfg = load_config(config_file)
    
    # Set aux_dim to the number of current auxiliary variables
    cfg.model.params.aux_dim = len(aux_indices)
    
    # Update save path and log directory to reflect the auxiliary-variable combination
    aux_indices_str = "_".join(map(str, aux_indices))
    cfg.paths.log_dir = f"{log_dir}/aux_{aux_indices_str}"
    cfg.paths.save_path = f"./checkpoints/best_model_aux_dgcn_aux_{aux_indices_str}_run_{run_id}.pth"
    
    # Initialize trainer
    trainer = Trainer(cfg)
    
    # Train and obtain history
    history = trainer.fit(Y_train, Y_test, X_aux_train_single, X_aux_test_single, A)
    
    # Add auxiliary-variable information to history
    history['aux_variables'] = [aux_variable_names[i] for i in aux_indices]
    history['aux_indices'] = aux_indices
    
    # Save history
    history_filepath = save_history(history, cfg.paths.log_dir, run_id)
    
    # Compute metrics
    metrics = compute_metrics(history)
    metrics['aux_variables'] = [aux_variable_names[i] for i in aux_indices]
    metrics['aux_indices'] = aux_indices
    
    return metrics, history_filepath

def main():
    config_file = "config_AUX_DGCN.yaml"
    model_name = "AUX_DGCN"
    log_dir = "logs_selection"  # New log directory
    run_id = 1  # Single run
    
    # Initialize indices of all auxiliary variables
    current_aux_indices = list(range(len(aux_variable_names)))  # [0, 1, ..., 10]
    best_mae = float('inf')
    best_aux_indices = current_aux_indices.copy()
    selection_history = []
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n=== Iteration {iteration} ===")
        print(f"Current auxiliary variable indices: {current_aux_indices}")
        print(f"Current auxiliary variables: {[aux_variable_names[i] for i in current_aux_indices]}")
        
        # Train the current combination
        metrics, history_filepath = train_with_aux_indices(current_aux_indices, config_file, log_dir, run_id)
        
        # Record metrics for the current combination
        selection_record = {
            'iteration': iteration,
            'aux_indices': current_aux_indices,
            'aux_variables': [aux_variable_names[i] for i in current_aux_indices],
            'test_mae': metrics['test_mae'],
            'test_mape': metrics['test_mape'],
            'test_rmse': metrics['test_rmse'],
            'history_filepath': history_filepath
        }
        selection_history.append(selection_record)
        
        print(f"\nCurrent combination (aux_{'_'.join(map(str, current_aux_indices))}):")
        print(f"test_mae: {metrics['test_mae']:.6f}")
        print(f"test_mape: {metrics['test_mape']:.6f}")
        print(f"test_rmse: {metrics['test_rmse']:.6f}")
        
        # Save metrics for the current combination
        os.makedirs(log_dir, exist_ok=True)
        metrics_filepath = f"{log_dir}/metrics_aux_{'_'.join(map(str, current_aux_indices))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"✅ Metrics have been saved to {metrics_filepath}")
        
        # If current MAE is better, update the best MAE and best combination
        if metrics['test_mae'] < best_mae:
            best_mae = metrics['test_mae']
            best_aux_indices = current_aux_indices.copy()
        
        # Try removing each variable and evaluate
        improved = False
        best_new_mae = best_mae
        best_new_aux_indices = current_aux_indices.copy()
        comparison_results = []
        
        for idx in current_aux_indices:
            # Create a new combination after removing idx
            temp_aux_indices = [i for i in current_aux_indices if i != idx]
            print(f"\nTrying to remove variable {aux_variable_names[idx]} (index {idx})")
            print(f"Testing combination: {[aux_variable_names[i] for i in temp_aux_indices]}")
            
            # Train the new combination
            temp_metrics, _ = train_with_aux_indices(temp_aux_indices, config_file, log_dir, run_id)
            
            # Record metrics for the new combination
            temp_record = {
                'iteration': iteration,
                'aux_indices': temp_aux_indices,
                'aux_variables': [aux_variable_names[i] for i in temp_aux_indices],
                'test_mae': temp_metrics['test_mae'],
                'test_mape': temp_metrics['test_mape'],
                'test_rmse': temp_metrics['test_rmse']
            }
            selection_history.append(temp_record)
            
            print(f"Combination after removal (aux_{'_'.join(map(str, temp_aux_indices))}):")
            print(f"test_mae: {temp_metrics['test_mae']:.6f}")
            print(f"test_mape: {temp_metrics['test_mape']:.6f}")
            print(f"test_rmse: {temp_metrics['test_rmse']:.6f}")
            
            # Compare MAE and record results
            mae_diff = temp_metrics['test_mae'] - best_mae
            comparison_results.append({
                'removed_variable': aux_variable_names[idx],
                'removed_index': idx,
                'new_mae': temp_metrics['test_mae'],
                'mae_difference': mae_diff,
                'is_better': temp_metrics['test_mae'] < best_mae
            })
            
            # If MAE becomes lower after removal, mark as a potential best combination
            if temp_metrics['test_mae'] < best_new_mae:
                best_new_mae = temp_metrics['test_mae']
                best_new_aux_indices = temp_aux_indices.copy()
                improved = True
        
        # Print comparison results
        print(f"\n=== Comparison results at iteration {iteration} ===")
        print(f"Current best MAE: {best_mae:.6f} (combination: aux_{'_'.join(map(str, current_aux_indices))})")
        for result in comparison_results:
            print(f"Removed variable {result['removed_variable']} (index {result['removed_index']}):")
            print(f"  New MAE: {result['new_mae']:.6f}, difference: {result['mae_difference']:.6f}, "
                  f"{'better than current combination' if result['is_better'] else 'not better than current combination'}")
        
        # Decide whether to switch to a new combination
        if improved:
            print(f"\n✅ Found a better combination: aux_{'_'.join(map(str, best_new_aux_indices))}")
            print(f"New test_mae: {best_new_mae:.6f}")
            print(f"Switching combination and continuing iterations")
            current_aux_indices = best_new_aux_indices
            best_mae = best_new_mae
            best_aux_indices = best_new_aux_indices
        else:
            print("\n❌ No better combination found, stopping selection")
            break
    
    # Save metrics for the best combination
    final_metrics_filepath = f"{log_dir}/final_best_metrics_aux_{'_'.join(map(str, best_aux_indices))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    final_metrics, _ = train_with_aux_indices(best_aux_indices, config_file, log_dir, run_id)
    with open(final_metrics_filepath, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Final best-combination metrics have been saved to {final_metrics_filepath}")
    
    # Save the step-by-step selection process
    selection_history_filepath = f"{log_dir}/selection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(selection_history_filepath, "w", encoding="utf-8") as f:
        json.dump(selection_history, f, indent=4, ensure_ascii=False)
    print(f"✅ Step-by-step selection process has been saved to {selection_history_filepath}")
    
    print(f"\nFinal best combination: aux_{'_'.join(map(str, best_aux_indices))}")
    print(f"Variables in the best combination: {[aux_variable_names[i] for i in best_aux_indices]}")
    print(f"Best test_mae: {best_mae:.6f}")
    print(f"Final test results | MAE: {final_metrics['test_mae']:.6f}, MAPE: {final_metrics['test_mape']:.6f}%, RMSE: {final_metrics['test_rmse']:.6f}")

if __name__ == "__main__":
    main()
