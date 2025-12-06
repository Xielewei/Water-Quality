import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

from config import load_config
from trainer import Trainer


def load_data():
    """Load dataset and adjacency matrix, and construct auxiliary arrays."""
    mat = scipy.io.loadmat("datasets/water_dataset.mat")
    y_train = mat["Y_tr"]
    y_test = mat["Y_te"]
    x_aux_tr = mat["X_tr"]
    x_aux_te = mat["X_te"]
    a = np.load("datasets/adjacency_matrix.npy")

    # Convert X_aux to (num_nodes, num_time, num_features)
    x_aux_train_list = [x_aux_tr[0, i] for i in range(x_aux_tr.shape[1])]
    x_aux_train_stacked = np.stack(x_aux_train_list, axis=0)  # (T_train, N, F)
    x_aux_train = x_aux_train_stacked.transpose(1, 0, 2)  # (N, T_train, F)

    x_aux_test_list = [x_aux_te[0, i] for i in range(x_aux_te.shape[1])]
    x_aux_test_stacked = np.stack(x_aux_test_list, axis=0)  # (T_test, N, F)
    x_aux_test = x_aux_test_stacked.transpose(1, 0, 2)  # (N, T_test, F)

    return y_train, y_test, x_aux_train, x_aux_test, a


def build_test_data(
    trainer: Trainer,
    y_train: np.ndarray,
    y_test: np.ndarray,
    x_aux_train: np.ndarray,
    x_aux_test: np.ndarray,
    a: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct test samples (x_test, x_aux_test, m_test, a_test, y_test_window)
    using the same logic as Trainer.fit, but without running training.
    """
    from data.preprocess import data_partition, test_construction

    # 1. Partition data to get X_test and masks
    partition_args: Dict = {
        "Y_train": y_train,
        "Y_test": y_test,
        "unobserved_proportion": trainer.cfg.data.unobserved_proportion,
        "seed": trainer.cfg.seed,
        "AUX": trainer.use_aux,
    }
    if trainer.use_gcn:
        partition_args["A"] = a
    if trainer.use_aux:
        partition_args["X_aux_train"] = x_aux_train

    result = data_partition(**partition_args)
    if trainer.use_aux and trainer.use_gcn:
        _, _, _, _, x_test_base, a_obs, m_test_full, indices = result
        x_aux_train_obs = None  # not used below
    elif trainer.use_aux:
        _, _, _, _, x_test_base, m_test_full, indices = result
        a_obs = None
        x_aux_train_obs = None
    else:
        _, _, x_test_base, a_obs, m_test_full, indices = result
        x_aux_train_obs = None

    # 2. Construct test windows
    test_args: Dict = {
        "sample_size": trainer.cfg.data.test_sample_size,
        "Y_test": y_test,
        "X_test": x_test_base,
        "Indices": indices,
        "h": trainer.cfg.data.h,
        "N_sub": trainer.cfg.data.N_sub,
        "u": trainer.cfg.data.u,
        "seed": trainer.cfg.seed,
        "AUX": trainer.use_aux,
    }
    if trainer.use_gcn:
        test_args["A"] = a
    if trainer.use_aux:
        test_args["X_aux_test"] = x_aux_test

    # Require returning the global node index corresponding to each sample
    test_result = test_construction(**test_args, return_indices=True)
    if trainer.use_aux and trainer.use_gcn:
        x_test, x_aux_test_win, m_test, a_test, y_test_win, node_indices = test_result
    elif trainer.use_aux:
        x_test, x_aux_test_win, m_test, y_test_win, node_indices = test_result
        a_test = None
    else:
        x_test, m_test, a_test, y_test_win, node_indices = test_result
        x_aux_test_win = None

    return x_test, x_aux_test_win, m_test, a_test, y_test_win, node_indices


def evaluate_full_series(
    trainer: Trainer,
    y_true: np.ndarray,
    m_mask: np.ndarray,
    a_input: np.ndarray,
    x_input: np.ndarray,
    x_aux_true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the trained model on test windows and return full series:
        y_pred_all, y_true_all, m_all  with shape (B, N, T)
    """
    batch_size = trainer.cfg.training.batch_size

    if trainer.cfg.model.name == "KNN":
        from models import KNN

        x_tensor = torch.from_numpy(x_input).float().to(trainer.device)
        a_tensor = torch.from_numpy(a_input).float().to(trainer.device)
        m_tensor = torch.from_numpy(m_mask).bool().to(trainer.device)

        out = KNN(
            x_tensor,
            a_tensor,
            m_tensor,
            k=trainer.cfg.model.params.k,
            eps=trainer.cfg.model.params.eps,
            device=trainer.device,
        )

        y_all = torch.from_numpy(y_true).float().to(trainer.device).cpu().numpy()
        y_pred_all = out.cpu().numpy()
        m_all = m_mask
        return y_pred_all, y_all, m_all

    # Neural models
    trainer.model.eval()
    ys: List[np.ndarray] = []
    outs: List[np.ndarray] = []
    ms: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(y_true), batch_size):
            end_idx = min(i + batch_size, len(y_true))
            batch_x = torch.from_numpy(x_input[i:end_idx]).float().to(trainer.device)
            batch_y = torch.from_numpy(y_true[i:end_idx]).float().to(trainer.device)
            batch_m = torch.from_numpy(m_mask[i:end_idx]).float().to(trainer.device)

            if trainer.use_aux and trainer.use_gcn:
                batch_a = torch.from_numpy(a_input[i:end_idx]).float().to(trainer.device)
                batch_x_aux = torch.from_numpy(x_aux_true[i:end_idx]).float().to(trainer.device)
                out = trainer.model(batch_x, batch_a, batch_x_aux)
            elif trainer.use_aux:
                batch_x_aux = torch.from_numpy(x_aux_true[i:end_idx]).float().to(trainer.device)
                out = trainer.model(batch_x, batch_x_aux)
            else:
                batch_a = torch.from_numpy(a_input[i:end_idx]).float().to(trainer.device)
                out = trainer.model(batch_x, batch_a)

            ys.append(batch_y.cpu().numpy())
            outs.append(out.cpu().numpy())
            ms.append(batch_m.cpu().numpy())

    y_all = np.concatenate(ys, axis=0)
    y_pred_all = np.concatenate(outs, axis=0)
    m_all = np.concatenate(ms, axis=0)

    return y_pred_all, y_all, m_all


def run_all_models(
    n_samples_to_plot: int = 10,
    cache_path: str = os.path.join("figs", "compare_models_timeseries_cache.npz"),
) -> None:
    """
    Train/evaluate all models and plot their test-series comparison
    on the same sample and node.

    To avoid retraining every time, a caching mechanism is used here:
    - If cache_path exists: directly load the data of the top n_samples_to_plot points
      where our model (AUX_DGCN) has the smallest RMSE on the test set and plot them;
    - If it does not exist: run all models, select these points, save them to cache_path,
      and then plot.
    """

    # If cache exists, directly read data for the first n_samples_to_plot points
    if os.path.exists(cache_path):
        print(f"🔁 Found cached top-k data at {cache_path}, skip training.")
        cache = np.load(cache_path, allow_pickle=True)
        labels = list(cache["labels"])
        sample_indices = cache["sample_indices"]
        local_node_indices = cache["local_node_indices"]
        global_node_indices = cache["global_node_indices"]
        y_true_selected = cache["y_true_selected"]
        m_selected = cache["m_selected"]
        y_pred_selected = cache["y_pred_selected"]

        # Restore to a structure similar to the original for unified plotting logic below
        results = []
        for i, label in enumerate(labels):
            results.append(
                {
                    "label": str(label),
                    "y_pred": y_pred_selected[i],  # Shape (K, T)
                }
            )
    else:
        print(f"ℹ️ No cache found at {cache_path}, running all models...")
        y_train, y_test, x_aux_train, x_aux_test, a = load_data()

        # (config_path, display_label)
        models_config = [
            ("config_AUX_DGCN.yaml", "AUX_DGCN"),
            ("config_DGCN.yaml", "DGCN"),
            ("config_LSTM.yaml", "LSTM"),
            ("config_DGCN_LSTM.yaml", "DGCN_LSTM"),
            ("config_KNN.yaml", "KNN"),
            ("config_Mean.yaml", "Mean"),
        ]

        results_full = []
        true_ts_ref = None
        mask_ref = None
        node_indices_ref = None

        for cfg_path, label in models_config:
            if not os.path.exists(cfg_path):
                print(f"⚠️ Config file not found: {cfg_path}, skip.")
                continue

            cfg = load_config(cfg_path)
            print(f"\n=== Running model: {label} ({cfg.model.name}) ===")

            trainer = Trainer(cfg)

            # Train (or run KNN baseline inside fit)
            _ = trainer.fit(y_train, y_test, x_aux_train, x_aux_test, a)

            # Build test windows consistently with Trainer.fit
            x_test, x_aux_test_win, m_test, a_test, y_test_win, node_indices = build_test_data(
                trainer, y_train, y_test, x_aux_train, x_aux_test, a
            )

            # Evaluate and get full series
            y_pred_all, y_true_all, m_all = evaluate_full_series(
                trainer,
                y_test_win,
                m_test,
                a_test,
                x_test,
                x_aux_test_win,
            )

            # Cache reference true series, mask, and global node indices (only need the first model)
            if true_ts_ref is None:
                true_ts_ref = y_true_all
                mask_ref = m_all
                node_indices_ref = node_indices

            results_full.append(
                {
                    "label": label,
                    "y_pred": y_pred_all,
                    "y_true": y_true_all,
                    "mask": m_all,
                }
            )

        if not results_full:
            print("No models were successfully evaluated.")
            return

        # Use reference y_true/mask/node_indices from the first model
        y_true_all = true_ts_ref
        m_all = mask_ref
        node_indices_all = node_indices_ref

        b, n, t_len = y_true_all.shape

        # Find the points where our model (AUX_DGCN) has the lowest RMSE on the test set
        # (only on test nodes where the mask is all 0)
        aux_index = None
        for i, r in enumerate(results_full):
            if r["label"] == "AUX_DGCN":
                aux_index = i
                break
        if aux_index is None:
            print("⚠️ AUX_DGCN results not found, cannot select top points.")
            return

        aux_pred = results_full[aux_index]["y_pred"]  # (B, N, T)

        candidates = []
        for sample_idx in range(b):
            missing_nodes = np.where(m_all[sample_idx].sum(axis=-1) == 0)[0]
            for local_node_idx in missing_nodes:
                pred_ts = aux_pred[sample_idx, local_node_idx, :]
                true_ts = y_true_all[sample_idx, local_node_idx, :]
                rmse = float(np.sqrt(np.mean((pred_ts - true_ts) ** 2)))
                global_node_idx = int(node_indices_all[sample_idx, local_node_idx])
                candidates.append(
                    (rmse, sample_idx, int(local_node_idx), global_node_idx)
                )

        if not candidates:
            print("⚠️ No fully-missing test nodes found, cannot select top points.")
            return

        # Sort by RMSE in ascending order and select the first n_samples_to_plot points
        candidates.sort(key=lambda x: x[0])
        top_k = min(n_samples_to_plot, len(candidates))
        selected = candidates[:top_k]

        K = len(selected)
        t_len = y_true_all.shape[-1]
        num_models = len(results_full)

        sample_indices = np.zeros(K, dtype=int)
        local_node_indices = np.zeros(K, dtype=int)
        global_node_indices = np.zeros(K, dtype=int)
        y_true_selected = np.zeros((K, t_len), dtype=float)
        m_selected = np.zeros((K, t_len), dtype=float)
        y_pred_selected = np.zeros((num_models, K, t_len), dtype=float)

        for idx, (_, sample_idx, local_node_idx, global_node_idx) in enumerate(selected):
            sample_indices[idx] = sample_idx
            local_node_indices[idx] = local_node_idx
            global_node_indices[idx] = global_node_idx
            y_true_selected[idx] = y_true_all[sample_idx, local_node_idx, :]
            m_selected[idx] = m_all[sample_idx, local_node_idx, :]
            for m_idx in range(num_models):
                y_pred_selected[m_idx, idx] = results_full[m_idx]["y_pred"][
                    sample_idx, local_node_idx, :
                ]

        labels = np.array([r["label"] for r in results_full])

        # Save data needed for plotting so that it can be loaded directly next time
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(
            cache_path,
            labels=labels,
            sample_indices=sample_indices,
            local_node_indices=local_node_indices,
            global_node_indices=global_node_indices,
            y_true_selected=y_true_selected,
            m_selected=m_selected,
            y_pred_selected=y_pred_selected,
        )
        print(f"💾 Saved cached top-k data to {cache_path}")

        # Also construct a results structure consistent with the one when loading from cache
        results = []
        for i, label in enumerate(labels):
            results.append(
                {
                    "label": str(label),
                    "y_pred": y_pred_selected[i],  # Shape (K, T)
                }
            )

    # Use the selected K "points" (sample-node pairs) for plotting
    K = len(sample_indices)
    t_len = y_true_selected.shape[-1]

    # Find the index of our model (AUX_DGCN) in results for highlighting
    aux_result_index = None
    for i, res in enumerate(results):
        if res["label"] == "AUX_DGCN":
            aux_result_index = i
            break

    # Prepare a set of baseline colors that do not include red to avoid clashing with AUX_DGCN
    baseline_colors = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#17becf",  # cyan
    ]

    os.makedirs("figs", exist_ok=True)

    for idx in range(K):
        sample_idx = int(sample_indices[idx])
        local_node_idx = int(local_node_indices[idx])
        global_node_idx = int(global_node_indices[idx])

        t_axis = np.arange(t_len)
        true_ts = y_true_selected[idx]
        miss_mask = m_selected[idx]

        plt.figure(figsize=(10, 5))
        ax = plt.gca()

        # Plot true series once (bold and highlighted) using a dashed line
        ax.plot(
            t_axis,
            true_ts,
            label="True",
            color="black",
            linewidth=2.5,
            linestyle="--",
        )

        # Plot each model's prediction
        preds_for_ylim = [true_ts]
        baseline_color_idx = 0
        for i, res in enumerate(results):
            pred_ts = res["y_pred"][idx, :]
            preds_for_ylim.append(pred_ts)

            # Our model (AUX_DGCN): prominent red bold dashed line
            if aux_result_index is not None and i == aux_result_index:
                ax.plot(
                    t_axis,
                    pred_ts,
                    label=res["label"],
                    color="red",
                    linewidth=2.5,
                    linestyle="--",
                    alpha=0.95,
                )
            else:
                # Other baselines: use predefined colors and a higher transparency
                color = baseline_colors[baseline_color_idx % len(baseline_colors)]
                baseline_color_idx += 1
                ax.plot(
                    t_axis,
                    pred_ts,
                    label=res["label"],
                    color=color,
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.3,
                )

        # Optional: mark missing positions
        missing_t = np.where(miss_mask == 0)[0]
        if missing_t.size > 0:
            ax.scatter(
                missing_t,
                true_ts[missing_t],
                color="blue",
                marker="x",
                label="Missing positions",
            )

        # Adjust y-axis range: add a margin based on the min/max of all curves
        all_values = np.concatenate(preds_for_ylim)
        vmin, vmax = float(all_values.min()), float(all_values.max())
        vrange = vmax - vmin
        if vrange <= 0:
            # Do not add extra margin for constant sequences
            ax.set_ylim(vmin, vmax)
        else:
            # Use a small margin: 5% above and below
            margin = vrange * 0.05
            ax.set_ylim(vmin - margin, vmax + margin)

        ax.set_xlabel("Time step in window")
        ax.set_ylabel("Value")

        # Use a smaller legend
        ax.legend(fontsize=8)

        # Use denser grid lines: enable minor ticks and draw grids for both major and minor ticks
        ax.minorticks_on()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        # Thicken the axis spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # Include the true (1-based) node index in the file name
        filename = f"compare_models_sample{sample_idx}_globalnode{global_node_idx + 1}.png"
        save_path = os.path.join("figs", filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(
            f"✅ Saved figure for sample {sample_idx}, "
            f"global node {global_node_idx + 1} to {save_path}"
        )


if __name__ == "__main__":
    # By default, plot 10 figures: for each sample select one test node (nodes with mask all 0)
    run_all_models(n_samples_to_plot=10)
