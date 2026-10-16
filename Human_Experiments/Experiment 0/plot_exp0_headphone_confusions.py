"""
Plot Experiment 0 confusion matrices split by headphone type and stimulus condition.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(ROOT, "Figures", "all_figs")

MATERIALS = ["Metal", "Glass", "Ceramic", "Stone", "Wood", "Plastic", "Cardboard"]
MATERIAL_TO_IDX = {m.lower(): i for i, m in enumerate(MATERIALS)}

CONDITION_LABELS = {
    0: "Recorded sounds",
    1: "Resynthesized IRs",
    2: "Full generative model",
}

GROUP_LABELS = {
    "in ear": "In-ear",
    "over the ear": "Over-the-ear",
}


def load_headphones():
    path = os.path.join(ROOT, "participant_data", "Exp0_metadata_with_headphones.csv")
    groups = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            groups.append(row["Headphones"].strip().lower())
    return np.array(groups)


def build_confusion_matrix(df, condition, participant_mask=None):
    matrix = np.zeros((len(MATERIALS), len(MATERIALS)), dtype=int)
    trial_indices = [
        int(column.split("_")[1])
        for column in df.columns
        if column.endswith("_condition")
    ]

    for participant_idx, row in df.iterrows():
        if participant_mask is not None and not participant_mask[participant_idx]:
            continue
        for trial_idx in trial_indices:
            if int(row[f"trial_{trial_idx}_condition"]) != condition:
                continue
            true_mat = str(row[f"trial_{trial_idx}_true"]).lower()
            resp_mat = str(row[f"trial_{trial_idx}_response"]).lower()
            if true_mat in MATERIAL_TO_IDX and resp_mat in MATERIAL_TO_IDX:
                matrix[MATERIAL_TO_IDX[true_mat], MATERIAL_TO_IDX[resp_mat]] += 1
    return matrix


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


def plot_exp0_headphone_confusions(df, headphones, save_path):
    cmap = create_white_to_blue_cmap()
    groups = ["in ear", "over the ear"]
    counts = {key: int(np.sum(headphones == key)) for key in groups}

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2))
    reference_matrices = {}

    for row_idx, group_key in enumerate(groups):
        mask = headphones == group_key
        group_cms = [
            build_confusion_matrix(df, cond, participant_mask=mask)
            for cond in range(3)
        ]
        reference_matrices[group_key] = group_cms[0]

        for col_idx, (cm, cond) in enumerate(zip(group_cms, range(3))):
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
                ax.set_title(CONDITION_LABELS[cond], fontsize=11, fontweight="bold", pad=8)
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
                    fontsize=9,
                    color="0.25",
                )

    fig.suptitle(
        "Experiment 0: Confusion matrices by headphone type",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(0.5, 0.03, "Response", ha="center", fontsize=11)
    fig.text(0.02, 0.5, "True material", va="center", rotation="vertical", fontsize=11)

    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.62])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=0.5))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Row-normalized proportion")

    fig.subplots_adjust(left=0.08, right=0.9, top=0.9, bottom=0.1, wspace=0.08, hspace=0.12)
    fig.savefig(save_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data_path = os.path.join(SCRIPT_DIR, "Experiment_0_data.csv")
    df = pd.read_csv(data_path)
    headphones = load_headphones()
    if len(df) != len(headphones):
        raise ValueError(f"Participant count mismatch: data={len(df)}, metadata={len(headphones)}")

    output_path = os.path.join(OUT_DIR, "Headphone_comparison_Exp0_confusion_matrices.svg")
    plot_exp0_headphone_confusions(df, headphones, output_path)


if __name__ == "__main__":
    main()
