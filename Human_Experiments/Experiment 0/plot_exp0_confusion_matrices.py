"""
Analysis script for Experiment 0.
Reads the pre-optimized CSV (one row per participant) and plots confusion matrices as SVG.
No modifications are applied - data is plotted as-is.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define materials in a consistent order
MATERIALS = ['Metal', 'Glass', 'Ceramic', 'Stone', 'Wood', 'Plastic', 'Cardboard']
MATERIAL_TO_IDX = {m.lower(): i for i, m in enumerate(MATERIALS)}

# Condition labels
CONDITION_LABELS = {
    0: 'Recorded sounds',
    1: 'Resynthesized IRs',
    2: 'Full generative model'
}


def build_confusion_matrix_from_wide(df, condition):
    """Build a confusion matrix from wide-format participant data."""
    n_materials = len(MATERIALS)
    matrix = np.zeros((n_materials, n_materials), dtype=int)
    
    # Find all trial columns
    trial_cols = [c for c in df.columns if c.startswith('trial_') and c.endswith('_condition')]
    trial_indices = [int(c.split('_')[1]) for c in trial_cols]
    
    for trial_idx in trial_indices:
        true_col = f'trial_{trial_idx}_true'
        resp_col = f'trial_{trial_idx}_response'
        cond_col = f'trial_{trial_idx}_condition'
        
        if true_col not in df.columns or resp_col not in df.columns:
            continue
        
        for _, row in df.iterrows():
            if row[cond_col] != condition:
                continue
            
            true_mat = str(row[true_col]).lower()
            resp_mat = str(row[resp_col]).lower()
            
            if true_mat in MATERIAL_TO_IDX and resp_mat in MATERIAL_TO_IDX:
                true_idx = MATERIAL_TO_IDX[true_mat]
                resp_idx = MATERIAL_TO_IDX[resp_mat]
                matrix[true_idx, resp_idx] += 1
    
    return matrix


def normalize_confusion_matrix(cm):
    """Normalize confusion matrix by row."""
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / np.where(row_sums > 0, row_sums, 1)
    return np.nan_to_num(cm_norm, nan=0.0)


def create_white_to_blue_cmap():
    """Create a custom white-to-blue colormap matching the MATLAB style."""
    colors = [(1, 1, 1), (0, 0, 1)]  # White to blue
    return LinearSegmentedColormap.from_list('white_blue', colors, N=64)


def plot_three_confusion_matrices(cm_list, materials, condition_labels, 
                                   n_participants=0, save_path=None):
    """
    Plot three confusion matrices side by side in MATLAB style.
    Uses white=0, blue=max color scheme with clim [0, 0.5].
    Values are normalized by row (each row sums to 1).
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    
    # Create custom colormap
    cmap = create_white_to_blue_cmap()
    
    # Normalize matrices by row
    cm_norms = []
    for cm in cm_list:
        cm_norm = normalize_confusion_matrix(cm)
        cm_norms.append(cm_norm)
    
    n = len(materials)
    
    # Calculate cross-correlations with the first (recording) condition
    correlations = []
    for i, cm_norm in enumerate(cm_norms):
        if i == 0:
            correlations.append(1.0)
        else:
            if np.std(cm_norms[0].flatten()) > 0 and np.std(cm_norm.flatten()) > 0:
                corr = np.corrcoef(cm_norms[0].flatten(), cm_norm.flatten())[0, 1]
            else:
                corr = 0.0
            correlations.append(corr)
    
    for idx, (ax, cm_norm, cm_raw, title) in enumerate(zip(
        axes, cm_norms, cm_list, condition_labels
    )):
        # Create heatmap with white=0, blue=max, clim [0, 0.5]
        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=0.5, aspect='equal')
        
        # Show labels on first subplot only
        if idx == 0:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([m.title() for m in materials], fontsize=10, rotation=45, ha='right')
            ax.set_yticklabels([m.title() for m in materials], fontsize=10)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.set_title(f'{title}', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    plt.close(fig)
    
    return correlations


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "Experiment_0_data.csv")
    output_svg = os.path.join(script_dir, "Experiment_0_confusion_matrices.svg")
    
    df = pd.read_csv(input_csv)
    n_participants = len(df)
    
    # Count trials per condition
    trial_cols = [c for c in df.columns if c.endswith('_condition')]
    n_trials = len(trial_cols)
    print(f"Trials per participant: {n_trials}")
    
    # Count trials per condition across all participants
    for cond in [0, 1, 2]:
        count = 0
        for col in trial_cols:
            count += (df[col] == cond).sum()
        print(f"  Condition {cond} ({CONDITION_LABELS[cond]}): {count} trials")
    
    # Build confusion matrices for each condition
    print("\nBuilding confusion matrices...")
    cm_recording = build_confusion_matrix_from_wide(df, 0)
    cm_mode_sampling = build_confusion_matrix_from_wide(df, 1)
    cm_full_model = build_confusion_matrix_from_wide(df, 2)
    
    # Print summary statistics
    print("\n" + "-" * 50)
    for cm, label in [(cm_recording, "Recording"), 
                      (cm_mode_sampling, "Mode sampling"),
                      (cm_full_model, "Full model")]:
        print(f"\n{label} condition confusion matrix:")
        total_correct = 0
        total_trials = 0
        for i, mat in enumerate(MATERIALS):
            row_sum = cm[i].sum()
            correct = cm[i, i]
            acc = correct / row_sum if row_sum > 0 else 0
            total_correct += correct
            total_trials += row_sum
            print(f"  {mat:12s}: {acc:.1%} ({correct}/{row_sum})")
        overall_acc = total_correct / total_trials if total_trials > 0 else 0
        print(f"  {'Overall':12s}: {overall_acc:.1%} ({total_correct}/{total_trials})")
    
    # Print demographics summary
    print("\n" + "-" * 50)
    print("Demographics:")
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        print(f"  Gender: {dict(gender_counts)}")
    if 'age' in df.columns:
        print(f"  Age: mean={df['age'].mean():.1f}, SD={df['age'].std():.1f}")
    
    # Plot confusion matrices
    condition_labels = [
        CONDITION_LABELS[0],
        CONDITION_LABELS[1],
        CONDITION_LABELS[2],
    ]
    
    print("\nGenerating SVG plot...")
    correlations = plot_three_confusion_matrices(
        [cm_recording, cm_mode_sampling, cm_full_model],
        MATERIALS,
        condition_labels,
        n_participants=n_participants,
        save_path=output_svg
    )
    
    # Print correlation summary
    print("\n" + "-" * 50)
    print("Correlations with Recording condition:")
    for i, (label, corr) in enumerate(zip(condition_labels, correlations)):
        if i > 0:
            print(f"  {label}: r = {corr:.3f}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
