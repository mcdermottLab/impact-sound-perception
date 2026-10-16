"""
Compare in-ear vs over-the-ear headphone performance for Experiments 0, 1, 3a, 3b, 3c and 4.

Each experiment is plotted using the same plot type as its original (non-headphone-split)
figure:
  - Exp0: grouped bar chart of accuracy by stimulus condition
  - Exp1: grouped bar chart of accuracy by model condition
  - Exp3a/3b/3c: errorbar line plot of accuracy vs. mass ratio (recorded-sound condition)
  - Exp4: errorbar line plot of d' vs. damping ratio (recorded-sound condition)

Experiment 2's original plot type (confusion matrices) is handled separately in
`Human_Experiments/Experiment 2/plot_exp2_headphone_confusions.py`, mirroring
`Human_Experiments/Experiment 0/plot_exp0_headphone_confusions.py`.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "Figures", "all_figs")

MATERIALS = ["metal", "glass", "ceramic", "stone", "wood", "plastic", "cardboard"]
MAT_IDX = {m: i for i, m in enumerate(MATERIALS)}

CONDITION_LABELS = {
    0: "Recorded sounds",
    1: "Resynthesized IRs",
    2: "Full generative model",
}

MASS_RATIOS = np.array([1.8, 3.16, 10, 31.6, 100])
MASS_LABELS = ["1.8", "3.16", "10", "31.6", "100"]

DAMPING_RATIOS = np.array([0.25, 0.5, 0.75, 0.90])
DAMPING_LABELS = ["0.25", "0.5", "0.75", "0.90"]

EXP1_CONDITION_LABELS = [
    "Full generative\nmodel",
    "Inverted\ncovariances",
    "2 modes\nGaussian",
    "Scaled\nvariances",
    "Uniform\nmodel",
    "Inverted\ndistributions",
    "Linear\ndecay",
]

GROUP_COLORS = {
    "in ear": "#2166ac",
    "over the ear": "#b2182b",
}
GROUP_LABELS = {
    "in ear": "In-ear",
    "over the ear": "Over-the-ear",
}
GROUPS = ["in ear", "over the ear"]


def load_headphones(exp):
    groups = []
    all_path = os.path.join(ROOT, "participant_data", f"{exp}_metadata_all.csv")
    if os.path.exists(all_path):
        paths = [all_path]
    else:
        paths = [
            os.path.join(ROOT, "participant_data", f"{exp}_metadata_with_headphones.csv"),
            os.path.join(ROOT, "participant_data", f"{exp}_metadata_headphone_test.csv"),
        ]
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                groups.append(row["Headphones"].strip().lower())
    return np.array(groups)


def group_counts(headphones):
    return {GROUP_LABELS[key]: int(np.sum(headphones == key)) for key in GROUPS}


def build_confusion_matrix(true_labels, resp_labels):
    cm = np.zeros((7, 7), dtype=int)
    for true_mat, resp_mat in zip(true_labels, resp_labels):
        true_idx = MAT_IDX.get(str(true_mat).lower(), -1)
        resp_idx = MAT_IDX.get(str(resp_mat).lower(), -1)
        if true_idx >= 0 and resp_idx >= 0:
            cm[true_idx, resp_idx] += 1
    return cm


def accuracy_from_labels(true_labels, resp_labels):
    cm = build_confusion_matrix(true_labels, resp_labels)
    return np.trace(cm) / cm.sum() if cm.sum() else np.nan


def participant_accuracy_by_condition(df, trial_indices):
    """Return shape (n_participants, 3) accuracies for conditions 0, 1, 2."""
    acc = np.zeros((len(df), 3))
    for participant_idx in range(len(df)):
        for cond in range(3):
            true_labels, resp_labels = [], []
            for trial_idx in trial_indices:
                if int(df.iloc[participant_idx][f"trial_{trial_idx}_condition"]) != cond:
                    continue
                true_labels.append(df.iloc[participant_idx][f"trial_{trial_idx}_true"])
                resp_labels.append(df.iloc[participant_idx][f"trial_{trial_idx}_response"])
            acc[participant_idx, cond] = accuracy_from_labels(true_labels, resp_labels)
    return acc


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))


def plot_grouped_bars(ax, x, group_data, ylabel, title, n_conditions, xticklabels, ylim):
    width = 0.34
    offsets = [-width / 2, width / 2]
    for offset, group_key in zip(offsets, GROUPS):
        means = [mean_sem(group_data[group_key][cond])[0] for cond in range(n_conditions)]
        sems = [mean_sem(group_data[group_key][cond])[1] for cond in range(n_conditions)]
        ax.bar(
            x + offset,
            means,
            width,
            yerr=sems,
            capsize=4,
            color=GROUP_COLORS[group_key],
            label=GROUP_LABELS[group_key],
            edgecolor="black",
            linewidth=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)


def plot_exp0(ax):
    data_path = os.path.join(ROOT, "Human_Experiments", "Experiment 0", "Experiment_0_data.csv")
    df = pd.read_csv(data_path)
    headphones = load_headphones("Exp0")
    trial_indices = [
        int(column.split("_")[1])
        for column in df.columns
        if column.endswith("_condition")
    ]

    acc = participant_accuracy_by_condition(df, trial_indices)
    group_data = {key: {cond: [] for cond in range(3)} for key in GROUPS}
    for participant_idx, group in enumerate(headphones):
        for cond in range(3):
            group_data[group][cond].append(acc[participant_idx, cond])

    counts = group_counts(headphones)
    plot_grouped_bars(
        ax,
        np.arange(3),
        group_data,
        "Proportion correct",
        f"Experiment 0: Material categorization\n(n = {len(df)}; "
        f"in-ear = {counts['In-ear']}, over-the-ear = {counts['Over-the-ear']})",
        n_conditions=3,
        xticklabels=[CONDITION_LABELS[i] for i in range(3)],
        ylim=(0, 0.55),
    )


def plot_exp1(ax):
    data_path = os.path.join(ROOT, "Human_Experiments", "Experiment 1", "results_exp1.mat")
    mat = sio.loadmat(data_path)
    resp_cond = mat["resp_cond"]  # participants x 7 conditions x trials, 0/1 correct
    headphones = load_headphones("Exp1")

    partwise = resp_cond.mean(axis=2)  # participants x 7 conditions
    n_conditions = partwise.shape[1]

    group_data = {key: {cond: [] for cond in range(n_conditions)} for key in GROUPS}
    for participant_idx, group in enumerate(headphones):
        for cond in range(n_conditions):
            group_data[group][cond].append(partwise[participant_idx, cond])

    counts = group_counts(headphones)
    plot_grouped_bars(
        ax,
        np.arange(n_conditions),
        group_data,
        "Proportion correct",
        f"Experiment 1: Realism discrimination\n(n = {resp_cond.shape[0]}; "
        f"in-ear = {counts['In-ear']}, over-the-ear = {counts['Over-the-ear']})",
        n_conditions=n_conditions,
        xticklabels=EXP1_CONDITION_LABELS,
        ylim=(0.3, 1.0),
    )
    ax.axhline(0.5, linestyle="--", color="black", linewidth=1.5, alpha=0.7)
    ax.tick_params(axis="x", labelsize=8)


def plot_mass_discrimination(ax, exp_key, data_filename, title, n_participants_key="resp_cond", ylim=(0.45, 1.0), show_target_line=True):
    """Shared plotter for the mass-discrimination-style experiments (3a/3b/3c).

    Mirrors the errorbar-vs-mass-ratio figure ('figure(2)') produced by each
    experiment's own *_plot.m script, restricted to the recorded-sound condition
    (condition index 0), which is the condition most sensitive to headphone response.
    """
    exp_dir = {
        "Exp3a": "Experiment 3a",
        "Exp3b": "Experiment 3b",
        "Exp3c": "Experiment 3c",
    }[exp_key]
    data_path = os.path.join(ROOT, "Human_Experiments", exp_dir, data_filename)
    mat = sio.loadmat(data_path)
    resp_cond = mat["resp_cond"]  # participants x conditions x mass-ratios x trials
    headphones = load_headphones(exp_key)

    partwise = resp_cond.mean(axis=3)
    x = np.arange(len(MASS_RATIOS))

    for group_key in GROUPS:
        mask = headphones == group_key
        group_perf = partwise[mask]
        means = group_perf[:, 0, :].mean(axis=0)
        sems = group_perf[:, 0, :].std(axis=0, ddof=1) / np.sqrt(mask.sum())
        ax.errorbar(
            x,
            means,
            yerr=sems,
            marker="o",
            linewidth=2,
            capsize=4,
            color=GROUP_COLORS[group_key],
            label=GROUP_LABELS[group_key],
        )

    ax.axhline(0.5, linestyle="--", color="black", linewidth=1.5, alpha=0.7)
    if show_target_line:
        ax.axhline(0.8, linestyle=":", color="gray", linewidth=1.2, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MASS_LABELS)
    ax.set_xlabel("Mass ratio")
    ax.set_ylabel("Proportion correct")
    counts = group_counts(headphones)
    ax.set_title(
        f"{title}\n(n = {resp_cond.shape[0]}; in-ear = {counts['In-ear']}, "
        f"over-the-ear = {counts['Over-the-ear']})"
    )
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)


def plot_exp3a(ax):
    plot_mass_discrimination(
        ax,
        "Exp3a",
        "Exp3a_data.mat",
        "Experiment 3a: Mass discrimination",
        ylim=(0.45, 1.0),
    )


def plot_exp3b(ax):
    plot_mass_discrimination(
        ax,
        "Exp3b",
        "Exp3_data.mat",
        "Experiment 3b: Mass discrimination (recorded sounds)",
        ylim=(0.45, 1.0),
    )


def plot_exp3c(ax):
    plot_mass_discrimination(
        ax,
        "Exp3c",
        "Exp3c_data.mat",
        "Experiment 3c: Mass discrimination",
        ylim=(0.35, 1.0),
    )


def plot_exp4(ax):
    """d' vs. damping ratio for the recorded-sound condition, mirroring figure(2)
    in plot_exp4_results.m (errorbar line plot per condition), restricted to condition 0
    ('Recorded') and split by headphone type."""
    data_path = os.path.join(ROOT, "Human_Experiments", "Experiment 4", "Exp4_data.mat")
    mat = sio.loadmat(data_path)
    resp_cond = mat["resp_cond"]  # participants x 6 conditions x 5 damping-levels x trials
    headphones = load_headphones("Exp4")

    cond0 = resp_cond[:, 0, :, :].astype(float)  # participants x 5 x trials
    same_trials = cond0[:, 0, :]  # catch trials ("same"), participants x trials
    corr_same = (same_trials == 1).astype(float)
    damp_trials = cond0[:, 1:, :]  # participants x 4 damping-levels x trials
    corr_damp = np.isin(damp_trials, [2, 3]).astype(float)

    dp_max = 2.0
    x = np.arange(len(DAMPING_RATIOS))

    for group_key in GROUPS:
        mask = headphones == group_key
        cr_rate_part = corr_same[mask].mean(axis=1)
        fp_rate_part = np.clip(1 - cr_rate_part, 0.01, 0.99)
        hit_rate_part = np.clip(corr_damp[mask].mean(axis=2), 0.01, 0.99)

        d_prime_part = norm.ppf(hit_rate_part) - norm.ppf(fp_rate_part)[:, None]
        d_prime_part = np.clip(d_prime_part, None, dp_max)

        means = d_prime_part.mean(axis=0)
        sems = d_prime_part.std(axis=0, ddof=1) / np.sqrt(mask.sum())
        ax.errorbar(
            x,
            means,
            yerr=sems,
            marker="o",
            linewidth=2,
            capsize=4,
            color=GROUP_COLORS[group_key],
            label=GROUP_LABELS[group_key],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DAMPING_LABELS)
    ax.set_xlabel("Damping ratio")
    ax.set_ylabel("d'")
    counts = group_counts(headphones)
    ax.set_title(
        f"Experiment 4: Force/damping discrimination (recorded sounds)\n"
        f"(n = {resp_cond.shape[0]}; in-ear = {counts['In-ear']}, "
        f"over-the-ear = {counts['Over-the-ear']})"
    )
    ax.set_ylim(0, 2.4)
    ax.grid(axis="y", alpha=0.25)


PLOTTERS = {
    "Exp0": ("Headphone_comparison_Exp0.svg", plot_exp0, (5.2, 4)),
    "Exp1": ("Headphone_comparison_Exp1.svg", plot_exp1, (6.2, 4.4)),
    "Exp3a": ("Headphone_comparison_Exp3a.svg", plot_exp3a, (5.2, 4)),
    "Exp3b": ("Headphone_comparison_Exp3b.svg", plot_exp3b, (5.2, 4)),
    "Exp3c": ("Headphone_comparison_Exp3c.svg", plot_exp3c, (5.2, 4)),
    "Exp4": ("Headphone_comparison_Exp4.svg", plot_exp4, (5.2, 4)),
}

COMBINED_LAYOUT = [
    ["Exp0", "Exp1", "Exp3a"],
    ["Exp3b", "Exp3c", "Exp4"],
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    n_rows, n_cols = len(COMBINED_LAYOUT), len(COMBINED_LAYOUT[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.6 * n_rows))
    for row_idx, row in enumerate(COMBINED_LAYOUT):
        for col_idx, exp_key in enumerate(row):
            _, plot_fn, _ = PLOTTERS[exp_key]
            plot_fn(axes[row_idx, col_idx])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    combined_path = os.path.join(OUT_DIR, "Headphone_comparison_all.svg")
    fig.savefig(combined_path, bbox_inches="tight")
    print(f"Saved {combined_path}")
    plt.close(fig)

    for exp_key, (filename, plot_fn, figsize) in PLOTTERS.items():
        fig_i, ax_i = plt.subplots(figsize=figsize)
        plot_fn(ax_i)
        ax_i.legend(frameon=False)
        fig_i.tight_layout()
        out_path = os.path.join(OUT_DIR, filename)
        fig_i.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close(fig_i)


if __name__ == "__main__":
    main()
