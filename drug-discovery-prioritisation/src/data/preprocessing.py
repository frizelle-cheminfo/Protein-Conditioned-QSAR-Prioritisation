"""
Data preprocessing and cleaning.

Handles SMILES validation, pActivity conversion, and deduplication.
"""

import pandas as pd
import numpy as np
from rdkit import Chem


def clean_and_compute_pActivity(df):
    """
    Clean data and compute pActivity.

    Args:
        df (pd.DataFrame): Raw data with 'canonical_smiles' and 'standard_value' (in nM)

    Returns:
        pd.DataFrame: Cleaned data with pActivity column

    Conversion formula:
        pIC50 = 9 - log10(IC50_nM)
        Since IC50[M] = IC50[nM] * 1e-9
    """
    print("\nCleaning data...")
    print(f"  Start: {len(df)} records")

    # Drop missing
    df = df.dropna(subset=['canonical_smiles', 'standard_value'])
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df.dropna(subset=['standard_value'])

    # Compute pActivity
    df['pActivity'] = 9 - np.log10(df['standard_value'])

    # Filter outliers
    df = df[(df['pActivity'] >= 3) & (df['pActivity'] <= 11)]

    # Validate SMILES
    valid_data = []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if mol:
            canonical = Chem.MolToSmiles(mol)
            valid_data.append({
                'smiles': canonical,
                'pActivity': row['pActivity'],
                'activity_nM': row['standard_value'],
                'target_name': row['target_name'],
                'target_chembl_id': row.get('target_chembl_id', 'user')
            })

    df_clean = pd.DataFrame(valid_data)

    # Deduplicate (keep mean)
    df_clean = df_clean.groupby(['smiles', 'target_name']).agg({
        'pActivity': 'mean',
        'activity_nM': 'mean',
        'target_chembl_id': 'first'
    }).reset_index()

    print(f"  Final: {len(df_clean)} unique compound-target pairs")

    return df_clean
