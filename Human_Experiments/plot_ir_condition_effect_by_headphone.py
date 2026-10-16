"""
Does the paper's key IR-condition effect replicate within each headphone type?

The previous headphone-comparison scripts (plot_headphone_comparison.py,
Experiment 0/plot_exp0_headphone_confusions.py and Experiment 2/plot_exp2_headphone_confusions.py) show that *absolute*
performance is similar for in-ear vs. over-the-ear listeners. This script goes
one level deeper: for each experiment it isolates the specific IR/model-condition
comparison that the paper's own *_stats.m / Exp0_stats.py scripts treat as the key
result (e.g. "Full generative model" vs "Inverted covariances", or "Resynthesized
IRs" vs "Full generative model"), and re-runs that exact comparison separately
within the in-ear-only and over-the-ear-only subgroups.

Two kinds of output are produced:
  1. For the mass-ratio (Exp 3a/3b/3c) and damping-ratio (Exp 4) discrimination
     experiments: a two-panel figure (one panel per headphone group) showing all
     IR/model conditions, in the same style as the original *_plot.m figures.
     (Exp 0/1/2's original plot types already show every condition split by
     headphone type - grouped bars for Exp 1, confusion matrices for Exp 0/2 -
     so no new figure is needed there.)
  2. A printed summary table of the key test statistic (correlation, t-test, or
     interpolated threshold) and its significance, computed for the full sample
     and for each headphone subgroup separately.

Sample sizes for the over-the-ear subgroup are small (roughly 15-40 depending on
experiment), so p-values there are expected to be noisier / less powered than the
full-sample numbers - that's the point of looking.
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import norm, ttest_rel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from plot_headphone_comparison import (  # noqa: E402
    DAMPING_LABELS,
    DAMPING_RATIOS,
    GROUP_COLORS,
    GROUP_LABELS,
    GROUPS,
    MASS_LABELS,
    MASS_RATIOS,
    OUT_DIR,
    ROOT,
    load_headphones,
)

RNG = np.random.default_rng(0)

MASS_COND_NAMES = [
    "Recorded",
    "Full generative model",
    "Inverted covariances",
    "Scaled variances",
    "Uniform model",
    "Inverted distribution",
    "Linear decay",
]

EXP4_COND_NAMES = [
    "Recorded",
    "Full generative model",
    "Inverted covariance",
    "Scaled variance",
    "Uniform distribution",
    "Linear decay",
]

EXP4_THRESHOLD_LABELS = [
    "Recorded",
    "Full generative\nmodel",
    "Inverted\ncovariance",
    "Scaled\nvariance",
    "Uniform\ndistribution",
    "Linear\ndecay",
]

# Fixed condition palette used in the paper's exported Figures 9 and 10.
# Keep colors tied to condition index in every panel and experiment; do not
# restart with Matplotlib's default color cycle for headphone subgroups.
COND_COLORS = [
    "#1171be",  # Recorded
    "#dd5400",  # Full generative model
    "#edb120",  # Inverted covariance(s)
    "#8516d1",  # Scaled variance(s)
    "#3baa32",  # Uniform model/distribution
    "#2fbeef",  # Inverted distribution (Exp 3) / linear decay (Exp 4)
    "#d1048b",  # Linear decay (Exp 3)
]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def group_masks(headphones):
    return {key: headphones == key for key in GROUPS}


def threshold_from_perf(perf, mass_ratios, target=0.8):
    """Mirrors the MATLAB `interp1(perf+tiny*(1:N), mass_ratios, target, 'linear','extrap')`."""
    tiny = np.arange(1, len(perf) + 1) * 1e-4
    x = perf + tiny
    order = np.argsort(x)
    return float(np.interp(target, x[order], mass_ratios[order]))


def swap_permute(data_ab, rng):
    """data_ab: (participants, 2, K, trials). Per (participant, trial), coin-flip
    which of the two condition slices (length K, e.g. mass-ratio levels) goes to
    slot 0 vs slot 1 - matches the *_stats.m permutation procedure."""
    n_participants, _, _, n_trials = data_ab.shape
    flip = rng.integers(0, 2, size=(n_participants, 1, n_trials))
    d0, d1 = data_ab[:, 0], data_ab[:, 1]
    perm0 = np.where(flip == 0, d0, d1)
    perm1 = np.where(flip == 0, d1, d0)
    return perm0, perm1


# --------------------------------------------------------------------------
# Experiment 0: Resynthesized IRs vs Full generative model (corr. w/ Recorded)
# --------------------------------------------------------------------------

MATERIALS0 = ["metal", "glass", "ceramic", "stone", "wood", "plastic", "cardboard"]
MAT_IDX0 = {m: i for i, m in enumerate(MATERIALS0)}


def _load_exp0():
    df = pd.read_csv(os.path.join(ROOT, "Human_Experiments", "Experiment 0", "Experiment_0_data.csv"))
    trial_indices = [int(c.split("_")[1]) for c in df.columns if c.endswith("_condition")]
    true_mats = np.array([df[f"trial_{t}_true"].str.lower().values for t in trial_indices]).T
    resp_mats = np.array([df[f"trial_{t}_response"].str.lower().values for t in trial_indices]).T
    conds = np.array([df[f"trial_{t}_condition"].values for t in trial_indices]).T
    return true_mats, resp_mats, conds


def _cm_from_labels(true_labels, resp_labels):
    cm = np.zeros((7, 7), dtype=int)
    ti = np.array([MAT_IDX0.get(t, -1) for t in true_labels])
    ri = np.array([MAT_IDX0.get(r, -1) for r in resp_labels])
    valid = (ti >= 0) & (ri >= 0)
    np.add.at(cm, (ti[valid], ri[valid]), 1)
    return cm


def exp0_condition_effect(mask, n_perm=5000):
    true_mats, resp_mats, conds = _load_exp0()
    true_mats, resp_mats, conds = true_mats[mask], resp_mats[mask], conds[mask]
    n_participants = true_mats.shape[0]

    cm_baseline = _cm_from_labels(true_mats[conds == 0], resp_mats[conds == 0])

    part_cm1, part_cm2 = [], []
    for p in range(n_participants):
        m1 = conds[p] == 1
        m2 = conds[p] == 2
        part_cm1.append(_cm_from_labels(true_mats[p, m1], resp_mats[p, m1]))
        part_cm2.append(_cm_from_labels(true_mats[p, m2], resp_mats[p, m2]))
    part_cm1, part_cm2 = np.array(part_cm1), np.array(part_cm2)
    cm1, cm2 = part_cm1.sum(axis=0), part_cm2.sum(axis=0)

    corr_resynth = np.corrcoef(cm_baseline.flatten(), cm1.flatten())[0, 1]
    corr_full = np.corrcoef(cm_baseline.flatten(), cm2.flatten())[0, 1]
    actual_diff = corr_resynth - corr_full

    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        swap = RNG.integers(0, 2, size=n_participants)
        perm_cm1 = np.where(swap[:, None, None] == 0, part_cm1, part_cm2).sum(axis=0)
        perm_cm2 = np.where(swap[:, None, None] == 0, part_cm2, part_cm1).sum(axis=0)
        c1 = np.corrcoef(cm_baseline.flatten(), perm_cm1.flatten())[0, 1]
        c2 = np.corrcoef(cm_baseline.flatten(), perm_cm2.flatten())[0, 1]
        perm_diffs[i] = c1 - c2
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(actual_diff))
    return {
        "n": n_participants,
        "r_resynth": corr_resynth,
        "r_full_model": corr_full,
        "p_value": p_value,
    }


# --------------------------------------------------------------------------
# Experiment 1: Full generative model vs Inverted covariances (paired t-test)
# --------------------------------------------------------------------------

def exp1_condition_effect(mask):
    mat = sio.loadmat(os.path.join(ROOT, "Human_Experiments", "Experiment 1", "results_exp1.mat"))
    resp_cond = mat["resp_cond"]
    partwise = np.clip(resp_cond.mean(axis=2), None, 1.0)[mask]
    data1, data2 = partwise[:, 0], partwise[:, 1]
    t_stat, p_value = ttest_rel(data1, data2)
    pooled_std = np.sqrt((data1.std(ddof=1) ** 2 + data2.std(ddof=1) ** 2) / 2)
    cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else np.nan
    return {
        "n": partwise.shape[0],
        "mean_full_model": data1.mean(),
        "mean_inv_cov": data2.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
    }


# --------------------------------------------------------------------------
# Experiment 2: Full generative model vs Inverted covariances (confusion corr.)
# --------------------------------------------------------------------------

def exp2_condition_effect(mask, n_perm=5000):
    mat = sio.loadmat(os.path.join(ROOT, "Human_Experiments", "Experiment 2", "Expt2_data.mat"))
    key_part = mat["key_part"].ravel()
    matnum_part = mat["matnum_part"].ravel()
    resp_fin = mat["resp_fin"][mask]
    n_participants = resp_fin.shape[0]

    def build_cm(cond, participants_resp):
        idx = key_part == cond
        trues = matnum_part[idx]
        flat_true = np.tile(trues, participants_resp.shape[0])
        flat_resp = participants_resp[:, idx].ravel()
        cm = np.zeros((7, 7), dtype=int)
        np.add.at(cm, (flat_true - 1, flat_resp - 1), 1)
        return cm

    cm_baseline = build_cm(1, resp_fin)
    cm_full = build_cm(2, resp_fin)
    cm_inv = build_cm(3, resp_fin)
    corr_full = np.corrcoef(cm_baseline.flatten(), cm_full.flatten())[0, 1]
    corr_inv = np.corrcoef(cm_baseline.flatten(), cm_inv.flatten())[0, 1]
    actual_diff = corr_full - corr_inv

    idx_full = key_part == 2
    idx_inv = key_part == 3
    resp_full = resp_fin[:, idx_full]
    resp_inv = resp_fin[:, idx_inv]
    trues_full = matnum_part[idx_full]
    trues_inv = matnum_part[idx_inv]
    n_trials = resp_full.shape[1]

    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        flip = RNG.integers(0, 2, size=(n_participants, n_trials))
        perm_full = np.where(flip == 0, resp_full, resp_inv)
        perm_inv = np.where(flip == 0, resp_inv, resp_full)

        cm_perm_full = np.zeros((7, 7), dtype=int)
        np.add.at(cm_perm_full, (np.tile(trues_full, n_participants) - 1, perm_full.ravel() - 1), 1)
        cm_perm_inv = np.zeros((7, 7), dtype=int)
        np.add.at(cm_perm_inv, (np.tile(trues_inv, n_participants) - 1, perm_inv.ravel() - 1), 1)

        c1 = np.corrcoef(cm_baseline.flatten(), cm_perm_full.flatten())[0, 1]
        c2 = np.corrcoef(cm_baseline.flatten(), cm_perm_inv.flatten())[0, 1]
        perm_diffs[i] = c1 - c2

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(actual_diff))
    return {
        "n": n_participants,
        "r_full_model": corr_full,
        "r_inv_cov": corr_inv,
        "p_value": p_value,
    }


# --------------------------------------------------------------------------
# Experiments 3a/3b/3c: mass-ratio threshold (80% correct), Full model vs Inv. cov.
# --------------------------------------------------------------------------

MASS_DATA_FILES = {
    "Exp3a": ("Experiment 3a", "Exp3a_data.mat"),
    "Exp3b": ("Experiment 3b", "Exp3_data.mat"),
    "Exp3c": ("Experiment 3c", "Exp3c_data.mat"),
}


def _load_mass_resp_cond(exp_key):
    exp_dir, filename = MASS_DATA_FILES[exp_key]
    mat = sio.loadmat(os.path.join(ROOT, "Human_Experiments", exp_dir, filename))
    return mat["resp_cond"]


def mass_condition_effect(exp_key, mask, cond_a=1, cond_b=2, n_perm=3000):
    """cond_a, cond_b are 0-based condition indices (default: Full generative
    model vs Inverted covariances, matching each experiment's cond_names order)."""
    resp_cond = _load_mass_resp_cond(exp_key)[mask]
    n_participants = resp_cond.shape[0]

    perf_all = resp_cond.mean(axis=(0, 3))  # conditions x mass-ratio-levels
    threshold_a = threshold_from_perf(perf_all[cond_a], MASS_RATIOS)
    threshold_b = threshold_from_perf(perf_all[cond_b], MASS_RATIOS)
    actual_diff = threshold_b - threshold_a

    data_ab = resp_cond[:, [cond_a, cond_b], :, :]
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm_a, perm_b = swap_permute(data_ab, RNG)
        perf_a = perm_a.mean(axis=(0, 2))
        perf_b = perm_b.mean(axis=(0, 2))
        perm_diffs[i] = threshold_from_perf(perf_b, MASS_RATIOS) - threshold_from_perf(perf_a, MASS_RATIOS)

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(actual_diff))
    return {
        "n": n_participants,
        "threshold_full_model": threshold_a,
        "threshold_inv_cov": threshold_b,
        "p_value": p_value,
    }


def plot_mass_conditions_by_headphone(exp_key, ylim, title):
    resp_cond = _load_mass_resp_cond(exp_key)
    headphones = load_headphones(exp_key)
    masks = group_masks(headphones)
    x = np.arange(len(MASS_RATIOS))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, group_key in zip(axes, GROUPS):
        subset = resp_cond[masks[group_key]]
        partwise = subset.mean(axis=3)  # participants x conditions x sizes
        n_group = subset.shape[0]
        for cond_idx, cond_name in enumerate(MASS_COND_NAMES):
            means = partwise[:, cond_idx, :].mean(axis=0)
            sems = partwise[:, cond_idx, :].std(axis=0, ddof=1) / np.sqrt(n_group)
            ax.errorbar(
                x, means, yerr=sems, marker="o", markersize=4, linewidth=1.6, capsize=3,
                color=COND_COLORS[cond_idx], label=cond_name,
            )
        ax.axhline(0.5, linestyle="--", color="black", linewidth=1.3, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(MASS_LABELS)
        ax.set_xlabel("Mass ratio")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"{GROUP_LABELS[group_key]} (n = {n_group})", color=GROUP_COLORS[group_key], fontweight="bold")

    axes[0].set_ylabel("Proportion correct")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(title, y=1.15, fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"IRcondition_effect_by_headphone_{exp_key}.svg")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------
# Experiment 4: damping-ratio threshold (d'=1), Full model vs Inv. covariance
# --------------------------------------------------------------------------

def _dprime_from_resp(cond_data):
    """cond_data: participants x 5 (catch + 4 damping levels) x trials -> d' per damping level (4,)."""
    same_trials = cond_data[:, 0, :]
    corr_same = (same_trials == 1).astype(float)
    damp_trials = cond_data[:, 1:, :]
    corr_damp = np.isin(damp_trials, [2, 3]).astype(float)

    cr_rate = corr_same.mean()
    fp_rate = np.clip(1 - cr_rate, 0.01, 0.99)
    hit_rate = np.clip(corr_damp.mean(axis=(0, 2)), 0.01, 0.99)
    dp_max = norm.ppf(0.99)
    d_prime = np.clip(norm.ppf(hit_rate) - norm.ppf(fp_rate), None, dp_max)
    return d_prime


def _dprime_threshold(d_prime, damping_ratios, target=1.0):
    damps_fine = np.linspace(damping_ratios.min(), damping_ratios.max(), 1000)
    order = np.argsort(damping_ratios)
    interp = np.interp(damps_fine, damping_ratios[order], d_prime[order])
    return float(damps_fine[np.argmin(np.abs(interp - target))])


def exp4_condition_effect(mask, cond_a=1, cond_b=2, n_perm=3000):
    mat = sio.loadmat(os.path.join(ROOT, "Human_Experiments", "Experiment 4", "Exp4_data.mat"))
    resp_cond = mat["resp_cond"][mask]  # participants x conditions x 5 x trials
    n_participants = resp_cond.shape[0]

    dprime_a = _dprime_from_resp(resp_cond[:, cond_a, :, :])
    dprime_b = _dprime_from_resp(resp_cond[:, cond_b, :, :])
    threshold_a = _dprime_threshold(dprime_a, DAMPING_RATIOS)
    threshold_b = _dprime_threshold(dprime_b, DAMPING_RATIOS)
    actual_diff = threshold_b - threshold_a

    data_ab = resp_cond[:, [cond_a, cond_b], :, :]  # participants x 2 x 5 x trials
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm_a, perm_b = swap_permute(data_ab, RNG)
        dp_a = _dprime_from_resp(perm_a)
        dp_b = _dprime_from_resp(perm_b)
        perm_diffs[i] = _dprime_threshold(dp_b, DAMPING_RATIOS) - _dprime_threshold(dp_a, DAMPING_RATIOS)

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(actual_diff))
    return {
        "n": n_participants,
        "threshold_full_model": threshold_a,
        "threshold_inv_cov": threshold_b,
        "p_value": p_value,
    }


def plot_exp4_conditions_by_headphone():
    mat = sio.loadmat(os.path.join(ROOT, "Human_Experiments", "Experiment 4", "Exp4_data.mat"))
    resp_cond = mat["resp_cond"]
    headphones = load_headphones("Exp4")
    masks = group_masks(headphones)
    x = np.arange(len(DAMPING_RATIOS))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, group_key in zip(axes, GROUPS):
        subset = resp_cond[masks[group_key]]
        n_group = subset.shape[0]
        n_conditions = subset.shape[1]
        for cond_idx in range(n_conditions):
            cond_name = EXP4_COND_NAMES[cond_idx] if cond_idx < len(EXP4_COND_NAMES) else f"Condition {cond_idx}"
            d_prime_part = np.stack(
                [_dprime_from_resp(subset[p:p + 1, cond_idx, :, :]) for p in range(n_group)]
            )
            means = d_prime_part.mean(axis=0)
            sems = d_prime_part.std(axis=0, ddof=1) / np.sqrt(n_group)
            ax.errorbar(
                x, means, yerr=sems, marker="o", markersize=4, linewidth=1.6, capsize=3,
                color=COND_COLORS[cond_idx], label=cond_name,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(DAMPING_LABELS)
        ax.set_xlabel("Damping ratio")
        ax.set_ylim(0, 2.4)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"{GROUP_LABELS[group_key]} (n = {n_group})", color=GROUP_COLORS[group_key], fontweight="bold")

    axes[0].set_ylabel("d'")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.1))
    fig.suptitle("Experiment 4: Force/damping discrimination by IR condition", y=1.2, fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "IRcondition_effect_by_headphone_Exp4.svg")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_exp4_condition_thresholds_by_headphone(n_boot=10000):
    """Plot every IR-condition threshold in the style of paper Figure 10c.

    Headphone groups are shown in adjacent panels with a shared y-axis. As in
    Figure 10c, thresholds use gray bars and uncapped black bootstrap error
    bars; the error magnitude is the bootstrap standard deviation divided by
    two, matching the paper's plot_exp4_results.m.
    """
    mat = sio.loadmat(os.path.join(ROOT, "Human_Experiments", "Experiment 4", "Exp4_data.mat"))
    resp_cond = mat["resp_cond"]
    headphones = load_headphones("Exp4")
    masks = group_masks(headphones)
    rng = np.random.default_rng(20260821)
    x = np.arange(resp_cond.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
    for ax, group_key in zip(axes, GROUPS):
        subset = resp_cond[masks[group_key]]
        n_group = subset.shape[0]
        thresholds = np.empty(subset.shape[1])
        threshold_errors = np.empty(subset.shape[1])

        for cond_idx in range(subset.shape[1]):
            condition_data = subset[:, cond_idx, :, :]
            threshold = _dprime_threshold(_dprime_from_resp(condition_data), DAMPING_RATIOS)
            thresholds[cond_idx] = threshold

            boot_thresholds = np.empty(n_boot)
            for boot_idx in range(n_boot):
                participant_idx = rng.integers(0, n_group, size=n_group)
                boot_dprime = _dprime_from_resp(condition_data[participant_idx])
                boot_thresholds[boot_idx] = _dprime_threshold(boot_dprime, DAMPING_RATIOS)

            threshold_errors[cond_idx] = np.std(boot_thresholds, ddof=1) / 2

            condition_name = EXP4_COND_NAMES[cond_idx]
            print(
                f"  {GROUP_LABELS[group_key]:14s} n={n_group:3d}  "
                f"{condition_name:23s} threshold={threshold:.3f}  "
                f"bootstrap error={threshold_errors[cond_idx]:.3f}"
            )

        ax.bar(
            x,
            thresholds,
            width=0.8,
            color=(0.7, 0.7, 0.7),
            edgecolor="black",
            linewidth=0.8,
        )
        ax.errorbar(
            x,
            thresholds,
            yerr=threshold_errors,
            color="black",
            linestyle="none",
            linewidth=4,
            capsize=0,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(EXP4_COND_NAMES[: subset.shape[1]], rotation=45, ha="right")
        ax.set_xlabel("Condition")
        ax.set_ylim(0.3, 0.9)
        ax.set_title(f"{GROUP_LABELS[group_key]} (n = {n_group})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="both", labelsize=12)

    axes[0].set_ylabel("Damping ratio threshold")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "IRcondition_thresholds_by_headphone_Exp4.svg")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------
# Main: build all figures + print the subgroup-replication summary table
# --------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    print("Generating figures (all IR/model conditions, split by headphone type)...")
    plot_mass_conditions_by_headphone("Exp3a", ylim=(0.45, 1.0), title="Experiment 3a: Mass discrimination by IR condition")
    plot_mass_conditions_by_headphone("Exp3b", ylim=(0.45, 1.0), title="Experiment 3b: Mass discrimination by IR condition (recorded sounds)")
    plot_mass_conditions_by_headphone("Exp3c", ylim=(0.35, 1.0), title="Experiment 3c: Mass discrimination by IR condition")
    plot_exp4_conditions_by_headphone()
    print("\nExperiment 4 condition-wise thresholds by headphone type (participant-bootstrap CIs):")
    plot_exp4_condition_thresholds_by_headphone()

    print("\n" + "=" * 100)
    print("Key IR-condition effect, replicated per headphone subgroup")
    print("(full sample compared against the paper's own *_stats.m / Exp0_stats.py result)")
    print("=" * 100)

    headphones0 = load_headphones("Exp0")
    headphones1 = load_headphones("Exp1")
    headphones2 = load_headphones("Exp2")
    headphones3a = load_headphones("Exp3a")
    headphones3b = load_headphones("Exp3b")
    headphones3c = load_headphones("Exp3c")
    headphones4 = load_headphones("Exp4")

    def subgroup_masks(headphones):
        full = np.ones_like(headphones, dtype=bool)
        return {"Full sample": full, **{GROUP_LABELS[k]: headphones == k for k in GROUPS}}

    print("\n--- Experiment 0: Resynthesized IRs vs Full generative model (corr. with Recorded) ---")
    for label, mask in subgroup_masks(headphones0).items():
        r = exp0_condition_effect(mask)
        print(f"  {label:14s} n={r['n']:3d}  r_resynth={r['r_resynth']:.3f}  r_full_model={r['r_full_model']:.3f}  p={r['p_value']:.4f}")

    print("\n--- Experiment 1: Full generative model vs Inverted covariances (paired t-test) ---")
    for label, mask in subgroup_masks(headphones1).items():
        r = exp1_condition_effect(mask)
        print(f"  {label:14s} n={r['n']:3d}  mean_full={r['mean_full_model']:.3f}  mean_invcov={r['mean_inv_cov']:.3f}  t={r['t_stat']:.2f}  p={r['p_value']:.4f}  d={r['cohens_d']:.2f}")

    print("\n--- Experiment 2: Full generative model vs Inverted covariances (confusion corr.) ---")
    for label, mask in subgroup_masks(headphones2).items():
        r = exp2_condition_effect(mask)
        print(f"  {label:14s} n={r['n']:3d}  r_full_model={r['r_full_model']:.3f}  r_inv_cov={r['r_inv_cov']:.3f}  p={r['p_value']:.4f}")

    for exp_key, headphones in [("Exp3a", headphones3a), ("Exp3b", headphones3b), ("Exp3c", headphones3c)]:
        print(f"\n--- {exp_key}: mass-ratio threshold (80% correct), Full generative model vs Inverted covariances ---")
        for label, mask in subgroup_masks(headphones).items():
            r = mass_condition_effect(exp_key, mask)
            print(f"  {label:14s} n={r['n']:3d}  thresh_full={r['threshold_full_model']:.2f}  thresh_invcov={r['threshold_inv_cov']:.2f}  p={r['p_value']:.4f}")

    print("\n--- Experiment 4: damping-ratio threshold (d'=1), Full generative model vs Inverted covariance ---")
    for label, mask in subgroup_masks(headphones4).items():
        r = exp4_condition_effect(mask)
        print(f"  {label:14s} n={r['n']:3d}  thresh_full={r['threshold_full_model']:.3f}  thresh_invcov={r['threshold_inv_cov']:.3f}  p={r['p_value']:.4f}")

    print("\nNote: Exp 3a/3c do not have this comparison in their own *_stats.m scripts")
    print("(those only compute demographics) - it is applied here for consistency with Exp 3b/4.")


if __name__ == "__main__":
    main()
