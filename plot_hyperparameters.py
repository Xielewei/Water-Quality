import json
import os

import matplotlib.pyplot as plt
import numpy as np


# Global style for nicer aesthetics
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": False,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    }
)


def load_summary_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_param_groups(metrics: dict):
    h_keys = sorted(
        [k for k in metrics.keys() if k.startswith("h") and k[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    u_keys = sorted(
        [k for k in metrics.keys() if k.startswith("u") and k[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    nsub_keys = sorted(
        [k for k in metrics.keys() if k.startswith("N_sub")],
        key=lambda x: int(x.split("N_sub")[1]),
    )
    unobs_keys = sorted(
        [k for k in metrics.keys() if k.startswith("unobs_Nsub")],
        key=lambda x: float(x[len("unobs_Nsub") :].split("_Nsub")[0]),
    )
    return h_keys, u_keys, nsub_keys, unobs_keys


def _build_series(metrics: dict, keys, mode: str):
    xs = []
    mae_mean, mae_std = [], []
    mape_mean, mape_std = [], []

    for k in keys:
        if mode == "h":
            x = int(k[1:])
        elif mode == "u":
            x = int(k[1:])
        elif mode == "N_sub":
            x = int(k.split("N_sub")[1])
        elif mode == "unobs":
            x = float(k[len("unobs_Nsub") :].split("_Nsub")[0])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        xs.append(x)
        mae_mean.append(metrics[k]["test_mae_mean"])
        mae_std.append(metrics[k]["test_mae_std"])
        mape_mean.append(metrics[k]["test_mape_mean"])
        mape_std.append(metrics[k]["test_mape_std"])

    return xs, mae_mean, mae_std, mape_mean, mape_std


def _set_spine_width(ax, width: float = 1.5):
    for spine in ax.spines.values():
        spine.set_linewidth(width)


def _plot_param(
    ax,
    x,
    mae_mean,
    mae_std,
    mape_mean,
    mape_std,
    title: str,
    x_label: str,
):
    color_mae = "tab:blue"
    color_mape = "tab:red"

    x_arr = np.array(x, dtype=float)
    mae_mean_arr = np.array(mae_mean)
    mae_std_arr = np.array(mae_std)
    mape_mean_arr = np.array(mape_mean)
    mape_std_arr = np.array(mape_std)

    # MAE on left y-axis: line + shaded std
    ax.plot(
        x_arr,
        mae_mean_arr,
        "-o",
        color=color_mae,
        label="MAE (mean)",
        linewidth=2,
        markersize=5,
    )
    ax.fill_between(
        x_arr,
        mae_mean_arr - mae_std_arr,
        mae_mean_arr + mae_std_arr,
        color=color_mae,
        alpha=0.18,
        label="MAE ± std",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("MAE", color=color_mae)
    # Keep x-axis tick labels black and y-axis tick labels the same color as the left axis
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors=color_mae)
    # No subplot titles to keep figure clean
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)

    # Tighter but slightly expanded y-range to highlight robustness
    mae_min = float(np.min(mae_mean_arr - mae_std_arr))
    mae_max = float(np.max(mae_mean_arr + mae_std_arr))
    mae_range = max(mae_max - mae_min, 1e-4)
    margin = 0.3 * mae_range
    ax.set_ylim(max(0.0, mae_min - margin), mae_max + margin)

    # MAPE on right y-axis: line + shaded std
    ax2 = ax.twinx()
    ax2.plot(
        x_arr,
        mape_mean_arr,
        "-s",
        color=color_mape,
        label="MAPE (mean)",
        linewidth=2,
        markersize=5,
    )
    ax2.fill_between(
        x_arr,
        mape_mean_arr - mape_std_arr,
        mape_mean_arr + mape_std_arr,
        color=color_mape,
        alpha=0.18,
        label="MAPE ± std",
    )
    ax2.set_ylabel("MAPE (%)", color=color_mape)
    ax2.tick_params(axis="y", labelcolor=color_mape)

    mape_min = float(np.min(mape_mean_arr - mape_std_arr))
    mape_max = float(np.max(mape_mean_arr + mape_std_arr))
    mape_range = max(mape_max - mape_min, 1e-3)
    margin2 = 0.3 * mape_range
    ax2.set_ylim(mape_min - margin2, mape_max + margin2)

    # Make tick labels bold on both axes
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for label in ax2.get_yticklabels():
        label.set_fontweight("bold")

    # Spines thicker for both axes
    _set_spine_width(ax, width=1.6)
    _set_spine_width(ax2, width=1.6)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
    )


def plot_hyperparameters(summary_path: str, save_path: str):
    metrics = load_summary_metrics(summary_path)
    h_keys, u_keys, nsub_keys, unobs_keys = _extract_param_groups(metrics)

    # Wider, flatter layout for 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    ax_h, ax_u, ax_nsub, ax_unobs = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    if h_keys:
        x, mae_m, mae_s, mape_m, mape_s = _build_series(metrics, h_keys, "h")
        _plot_param(ax_h, x, mae_m, mae_s, mape_m, mape_s, title="Effect of h", x_label="h")

    if u_keys:
        x, mae_m, mae_s, mape_m, mape_s = _build_series(metrics, u_keys, "u")
        _plot_param(ax_u, x, mae_m, mae_s, mape_m, mape_s, title="Effect of u", x_label="u")

    if nsub_keys:
        x, mae_m, mae_s, mape_m, mape_s = _build_series(metrics, nsub_keys, "N_sub")
        _plot_param(
            ax_nsub,
            x,
            mae_m,
            mae_s,
            mape_m,
            mape_s,
            title="Effect of N_sub",
            x_label="N_sub",
        )

    if unobs_keys:
        x, mae_m, mae_s, mape_m, mape_s = _build_series(metrics, unobs_keys, "unobs")
        _plot_param(
            ax_unobs,
            x,
            mae_m,
            mae_s,
            mape_m,
            mape_s,
            title="Effect of unobserved proportion",
            x_label="U/(U+N)",
        )

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    summary_file = os.path.join("logs_parameter", "summary_metrics_20250927_193454.json")
    output_path = os.path.join("figs", "hyperparameters_mae_mape_2x2.png")
    plot_hyperparameters(summary_file, output_path)
    print(f"Saved figure to {output_path}")
