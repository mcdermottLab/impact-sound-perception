"""
Statistical analysis for Experiment 0.
Computes confusion matrices per condition, runs permutation tests to compare conditions,
and reports demographics.
"""

import os
import numpy as np
import pandas as pd

# Define materials in consistent order
MATERIALS = ['metal', 'glass', 'ceramic', 'stone', 'wood', 'plastic', 'cardboard']
MATERIAL_TO_IDX = {m: i for i, m in enumerate(MATERIALS)}

# Condition labels
CONDITION_LABELS = {
    0: 'Recorded sounds',
    1: 'Resynthesized IRs',
    2: 'Full generative model'
}


def load_data(filepath):
    """Load the experiment data and extract trial information."""
    df = pd.read_csv(filepath)
    
    # Find all trial columns
    trial_cols = [c for c in df.columns if c.endswith('_condition')]
    trial_indices = [int(c.split('_')[1]) for c in trial_cols]
    
    n_participants = len(df)
    n_trials = len(trial_indices)
    
    # Extract trial data into arrays
    true_materials = []
    responses = []
    conditions = []
    
    for trial_idx in trial_indices:
        true_col = f'trial_{trial_idx}_true'
        resp_col = f'trial_{trial_idx}_response'
        cond_col = f'trial_{trial_idx}_condition'
        
        true_materials.append(df[true_col].str.lower().values)
        responses.append(df[resp_col].str.lower().values)
        conditions.append(df[cond_col].values)
    
    # Convert to arrays: shape (n_participants, n_trials)
    true_materials = np.array(true_materials).T
    responses = np.array(responses).T
    conditions = np.array(conditions).T
    
    return df, true_materials, responses, conditions, n_participants, n_trials


def material_to_index(material):
    """Convert material name to index."""
    return MATERIAL_TO_IDX.get(material.lower(), -1)


def build_confusion_matrix(true_labels, pred_labels):
    """Build 7x7 confusion matrix from material labels."""
    n_materials = len(MATERIALS)
    cm = np.zeros((n_materials, n_materials), dtype=int)
    
    for true_mat, pred_mat in zip(true_labels, pred_labels):
        true_idx = material_to_index(true_mat)
        pred_idx = material_to_index(pred_mat)
        if true_idx >= 0 and pred_idx >= 0:
            cm[true_idx, pred_idx] += 1
    
    return cm


def get_conditionwise_responses(true_materials, responses, conditions, cond):
    """Get all true/response pairs for a specific condition."""
    mask = conditions == cond
    true_flat = true_materials[mask]
    resp_flat = responses[mask]
    return true_flat, resp_flat


def run_permutation_test(cm_baseline, resp_cond1, resp_cond2, matnum, n_participants, 
                         n_permutations=10000):
    """
    Run permutation test to compare correlations between two conditions.
    Tests if the difference in correlation with baseline is significant.
    """
    # Calculate actual correlations
    cm_cond1 = build_confusion_matrix(matnum, resp_cond1)
    cm_cond2 = build_confusion_matrix(matnum, resp_cond2)
    
    actual_corr1 = np.corrcoef(cm_baseline.flatten(), cm_cond1.flatten())[0, 1]
    actual_corr2 = np.corrcoef(cm_baseline.flatten(), cm_cond2.flatten())[0, 1]
    actual_diff = actual_corr1 - actual_corr2
    
    # Combine responses for permutation
    n_trials = len(resp_cond1)
    combined = np.stack([resp_cond1, resp_cond2], axis=0)  # shape: (2, n_trials)
    
    perm_diffs = []
    for _ in range(n_permutations):
        # Randomly swap responses between conditions for each trial
        perm_inds = np.random.randint(0, 2, size=n_trials)
        perm_resp1 = np.array([combined[perm_inds[i], i] for i in range(n_trials)])
        perm_resp2 = np.array([combined[1 - perm_inds[i], i] for i in range(n_trials)])
        
        # Build confusion matrices
        cm_perm1 = build_confusion_matrix(matnum, perm_resp1)
        cm_perm2 = build_confusion_matrix(matnum, perm_resp2)
        
        # Calculate correlation difference
        corr1 = np.corrcoef(cm_baseline.flatten(), cm_perm1.flatten())[0, 1]
        corr2 = np.corrcoef(cm_baseline.flatten(), cm_perm2.flatten())[0, 1]
        perm_diffs.append(corr1 - corr2)
    
    perm_diffs = np.array(perm_diffs)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(actual_diff))
    
    return p_value, actual_corr1, actual_corr2, actual_diff


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "Experiment_0_data.csv")
    
    # Load data
    df, true_materials, responses, conditions, n_participants, n_trials = load_data(data_file)
    
    print("=" * 70)
    print("Experiment 0 Statistical Analysis")
    print("=" * 70)
    print(f"\nParticipants: {n_participants}")
    print(f"Trials per participant: {n_trials}")
    
    # Build aggregate confusion matrices for each condition
    confusion_matrices = {}
    for cond in [0, 1, 2]:
        true_flat, resp_flat = get_conditionwise_responses(true_materials, responses, conditions, cond)
        cm = build_confusion_matrix(true_flat, resp_flat)
        confusion_matrices[cond] = cm
        print(f"\nCondition {cond} ({CONDITION_LABELS[cond]}): {len(true_flat)} trials")
    
    # Calculate correlations with the recorded sounds condition
    print("\n" + "-" * 70)
    print("Correlations with Recorded Sounds condition:")
    print("-" * 70)
    
    cm_baseline = confusion_matrices[0]
    for cond in [1, 2]:
        cm = confusion_matrices[cond]
        corr = np.corrcoef(cm_baseline.flatten(), cm.flatten())[0, 1]
        print(f"  {CONDITION_LABELS[cond]}: r = {corr:.3f}")
    
    # Permutation test between conditions 1 and 2
    # Get per-trial responses for each condition (flattened across participants)
    true_cond1, resp_cond1 = get_conditionwise_responses(true_materials, responses, conditions, 1)
    true_cond2, resp_cond2 = get_conditionwise_responses(true_materials, responses, conditions, 2)
    
    # Note: For a proper permutation test, we need matching trial pairs
    # Since trials are randomized per condition, we'll use the aggregate matrices approach
    
    print("\n" + "-" * 70)
    print("Permutation Test: Resynthesized IRs vs Full Generative Model")
    print("-" * 70)
    
    n_permutations = 10000
    
    # Calculate actual correlations
    corr_resynth = np.corrcoef(cm_baseline.flatten(), confusion_matrices[1].flatten())[0, 1]
    corr_full = np.corrcoef(cm_baseline.flatten(), confusion_matrices[2].flatten())[0, 1]
    actual_diff = corr_resynth - corr_full
    
    # Bootstrap permutation test using participant-level data
    # Build per-participant confusion matrices for each condition
    participant_cms = {0: [], 1: [], 2: []}
    
    for p_idx in range(n_participants):
        for cond in [0, 1, 2]:
            mask = conditions[p_idx, :] == cond
            if mask.sum() > 0:
                true_p = true_materials[p_idx, mask]
                resp_p = responses[p_idx, mask]
                cm_p = build_confusion_matrix(true_p, resp_p)
                participant_cms[cond].append(cm_p)
    
    # Aggregate per-participant CMs
    cms_cond1 = np.array(participant_cms[1])  # shape: (n_participants, 7, 7)
    cms_cond2 = np.array(participant_cms[2])  # shape: (n_participants, 7, 7)
    
    # Bootstrap with participant-level permutation
    perm_diffs = []
    for _ in range(n_permutations):
        # For each participant, randomly swap their condition 1 and 2 responses
        swap_inds = np.random.randint(0, 2, size=n_participants)
        
        perm_cm1 = np.zeros((7, 7))
        perm_cm2 = np.zeros((7, 7))
        
        for p_idx in range(n_participants):
            if swap_inds[p_idx] == 0:
                perm_cm1 += cms_cond1[p_idx]
                perm_cm2 += cms_cond2[p_idx]
            else:
                perm_cm1 += cms_cond2[p_idx]
                perm_cm2 += cms_cond1[p_idx]
        
        corr1 = np.corrcoef(cm_baseline.flatten(), perm_cm1.flatten())[0, 1]
        corr2 = np.corrcoef(cm_baseline.flatten(), perm_cm2.flatten())[0, 1]
        perm_diffs.append(corr1 - corr2)
    
    perm_diffs = np.array(perm_diffs)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(actual_diff))
    
    print(f"\nCorrelation (Resynthesized IRs with Recorded): r = {corr_resynth:.3f}")
    print(f"Correlation (Full Model with Recorded): r = {corr_full:.3f}")
    print(f"Difference: {actual_diff:.3f}")
    print(f"\nExperiment 0 -> p-value between Resynthesized IRs and Full Model is {p_value:.4f}")
    
    # Demographics
    print("\n" + "-" * 70)
    print("Demographics")
    print("-" * 70)
    
    if 'age' in df.columns:
        mean_age = df['age'].mean()
        std_age = df['age'].std()
        print(f"Participant age: Mean = {mean_age:.2f}, SD = {std_age:.2f}")
    
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        print(f"Gender distribution: {dict(gender_counts)}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
