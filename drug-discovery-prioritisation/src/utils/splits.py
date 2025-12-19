"""
Data splitting strategies.

Implements scaffold-based and leave-one-target-out splits.
"""

import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles):
    """
    Generate Murcko scaffold for a molecule.

    Args:
        smiles (str): SMILES string

    Returns:
        str: Scaffold SMILES or None if failed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None


def scaffold_split_within_target(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Scaffold split within each target.

    Groups molecules by Murcko scaffold and splits scaffolds (not molecules)
    into train/val/test sets. This ensures scaffolds don't leak between sets.

    Args:
        df (pd.DataFrame): DataFrame with 'target_name' and 'smiles' columns
        test_size (float): Fraction of data for test set
        val_size (float): Fraction of data for validation set
        random_state (int): Random seed

    Returns:
        dict: Mapping of target_name -> (train_idx, val_idx, test_idx)
    """
    print("\nScaffold split within targets...")

    splits = {}

    for target in df['target_name'].unique():
        target_df = df[df['target_name'] == target]
        indices = target_df.index.tolist()
        smiles_list = target_df['smiles'].tolist()

        # Scaffold grouping
        scaffold_to_idx = defaultdict(list)
        for i, smiles in zip(indices, smiles_list):
            scaffold = generate_scaffold(smiles)
            if scaffold:
                scaffold_to_idx[scaffold].append(i)

        scaffolds = list(scaffold_to_idx.items())
        np.random.seed(random_state)
        np.random.shuffle(scaffolds)

        n_total = len(indices)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)

        test_idx = []
        val_idx = []
        train_idx = []

        for scaffold, idx_list in scaffolds:
            if len(test_idx) < n_test:
                test_idx.extend(idx_list)
            elif len(val_idx) < n_val:
                val_idx.extend(idx_list)
            else:
                train_idx.extend(idx_list)

        splits[target] = (train_idx, val_idx, test_idx)

        print(f"  {target}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return splits


def leave_one_target_out_split(df):
    """
    Leave-one-target-out split.

    For each target, use it as test set and train on all others.
    Tests cross-target generalisation.

    Args:
        df (pd.DataFrame): DataFrame with 'target_name' column

    Returns:
        list: List of (train_idx, test_idx, test_target_name) tuples
    """
    print("\nLeave-one-target-out splits...")

    splits = []

    for test_target in df['target_name'].unique():
        train_idx = df[df['target_name'] != test_target].index.tolist()
        test_idx = df[df['target_name'] == test_target].index.tolist()

        splits.append((train_idx, test_idx, test_target))

        print(f"  Test={test_target}: train={len(train_idx)}, test={len(test_idx)}")

    return splits
