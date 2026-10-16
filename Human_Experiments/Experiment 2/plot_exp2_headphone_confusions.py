"""
Plot Experiment 2 confusion matrices split by headphone type and model condition.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(ROOT, "Figures", "all_figs")

MATERIALS = ["Metal", "Glass", "Ceramic", "Stone", "Wood", "Plastic", "Cardboard"]

CONDITION_LABELS = [
    "Recorded sounds",
    "Full generative model",
    "Inverted covariances",
    "Scaled variances",
    "2 modes Gaussian",
    "Uniform model",
    "Inverted distributions",
    "Linear decay",
]

GROUP_LABELS = {
    "in ear": "In-ear",
    "over the ear": "Over-the-ear",
}


def load_headphones():
    path = os.path.join(ROOT, "participant_data", "Exp2_metadata_with_headphones.csv")
    groups = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            groups.append(row["Headphones"].strip().lower())
    return np.array(groups)


def build_confusion_matrix(matnum_part, key_part, resp_sub, condition):
    """resp_sub: participants (in this group) x trials response matrix."""
    idx = key_part == condition
    trues = matnum_part[idx]
    n_participants = resp_sub.shape[0]
    flat_true = np.tile(trues, n_participants)
    flat_resp = resp_sub[:, idx].ravel()

    cm = np.zeros((7, 7), dtype=int)
    np.add.at(cm, (flat_true - 1, flat_resp - 1), 1)
    return cm


def normalize_confusion_matrix(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.nan_to_num(cm.astype(float) / np.where(row_sums > 0, row_sums, 1), nan=0.0)


def create_white_to_blue_cmap():
    return LinearSegmentedColormap.from_list("white_blue", [(1, 1, 1), (0, 0, 1)], N=64)


def correlation_with_reference(cm_ref, cm):
    ref = normalize_confusion_matrix(cm_ref).flatten()
    cur = normalize_confusion_matrix(cm).flatten()
    if np.std(ref) == 0 or np.std(cur) == 0:
        return np.nan
    return np.corrcoef(ref, cur)[0, 1]


def plot_exp2_headphone_confusions(mat, headphones, save_path):
    key_part = mat["key_part"].ravel()
    matnum_part = mat["matnum_part"].ravel()
    resp_fin = mat["resp_fin"]

    cmap = create_white_to_blue_cmap()
    groups = ["in ear", "over the ear"]
    counts = {key: int(np.sum(headphones == key)) for key in groups}
    n_conditions = len(CONDITION_LABELS)

    fig, axes = plt.subplots(2, n_conditions, figsize=(2.2 * n_conditions, 5.6))
    reference_matrices = {}

    for row_idx, group_key in enumerate(groups):
        mask = headphones == group_key
        resp_group = resp_fin[mask]
        group_cms = [
            build_confusion_matrix(matnum_part, key_part, resp_group, cond)
            for cond in range(1, n_conditions + 1)
        ]
        reference_matrices[group_key] = group_cms[0]

        for col_idx, cm in enumerate(group_cms):
            ax = axes[row_idx, col_idx]
            cm_norm = normalize_confusion_matrix(cm)

            ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=0.5, aspect="equal")

            if col_idx == 0:
                ax.set_yticks(range(len(MATERIALS)))
                ax.set_yticklabels(MATERIALS, fontsize=9)
            else:
                ax.set_yticks([])

            if row_idx == 1:
                ax.set_xticks(range(len(MATERIALS)))
                ax.set_xticklabels(MATERIALS, fontsize=9, rotation=45, ha="right")
            else:
                ax.set_xticks([])

            if row_idx == 0:
                ax.set_title(CONDITION_LABELS[col_idx], fontsize=9, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{GROUP_LABELS[group_key]}\n(n = {counts[group_key]})",
                    fontsize=11,
                    fontweight="bold",
                )

            if col_idx > 0:
                corr = correlation_with_reference(reference_matrices[group_key], cm)
                ax.text(
                    0.98,
                    0.02,
                    f"r = {corr:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="0.25",
                )

    fig.suptitle(
        "Experiment 2: Confusion matrices by headphone type",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(0.5, 0.02, "Response", ha="center", fontsize=11)
    fig.text(0.01, 0.5, "True material", va="center", rotation="vertical", fontsize=11)

    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.62])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=0.5))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Row-normalized proportion")

    fig.subplots_adjust(left=0.06, right=0.9, top=0.88, bottom=0.14, wspace=0.08, hspace=0.12)
    fig.savefig(save_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data_path = os.path.join(SCRIPT_DIR, "Expt2_data.mat")
    mat = sio.loadmat(data_path)
    headphones = load_headphones()
    if mat["resp_fin"].shape[0] != len(headphones):
        raise ValueError(
            f"Participant count mismatch: data={mat['resp_fin'].shape[0]}, metadata={len(headphones)}"
        )

    output_path = os.path.join(OUT_DIR, "Headphone_comparison_Exp2_confusion_matrices.svg")
    plot_exp2_headphone_confusions(mat, headphones, output_path)


if __name__ == "__main__":
    main()
