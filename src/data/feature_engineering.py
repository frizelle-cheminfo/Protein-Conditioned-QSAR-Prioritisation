"""
Feature generation for ligands and proteins.

Computes molecular fingerprints, descriptors, and combines with protein embeddings.
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors


def compute_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """
    Compute Morgan fingerprint for a molecule.

    Args:
        mol: RDKit molecule object
        radius (int): Morgan fingerprint radius
        n_bits (int): Number of bits in fingerprint

    Returns:
        np.ndarray: Binary fingerprint array
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_descriptors(mol):
    """
    Compute physicochemical descriptors for a molecule.

    Args:
        mol: RDKit molecule object

    Returns:
        np.ndarray: Array of 8 molecular descriptors
    """
    return np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Descriptors.TPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ], dtype=np.float32)


def _norm_key(x):
    """Normalise dictionary keys for robust lookup."""
    return str(x).strip().lower()


def generate_features_multitarget(
    df,
    target_embed_map,
    target_id_col="target_chembl_id",
    target_name_col="target_name",
    radius=2,
    n_bits=2048,
    verbose_every=500
):
    """
    Generate features for multi-target dataset.

    Args:
        df (pd.DataFrame): DataFrame with SMILES and target information
        target_embed_map (dict): Mapping of target names to protein embeddings
        target_id_col (str): Column name for target ChEMBL ID
        target_name_col (str): Column name for target name
        radius (int): Morgan fingerprint radius
        n_bits (int): Fingerprint size
        verbose_every (int): Print progress every N rows

    Returns:
        tuple: (X_ligand, X_protein_conditioned, mols, valid_indices)
            - X_ligand: Ligand-only features (fingerprint + descriptors)
            - X_protein_conditioned: Ligand features + protein embedding
            - mols: List of RDKit mol objects
            - valid_indices: Original DataFrame indices that were kept
    """
    # Build normalised lookup map
    norm_embed_map = {}
    for k, v in target_embed_map.items():
        norm_embed_map[_norm_key(k)] = np.asarray(v)

    # Diagnostics
    available_keys = list(norm_embed_map.keys())
    print("\n[DEBUG] target_embed_map keys (normalised) sample:", available_keys[:10])
    if target_id_col in df.columns:
        print("[DEBUG] df target_chembl_id sample:", df[target_id_col].dropna().astype(str).head(5).tolist())
    if target_name_col in df.columns:
        print("[DEBUG] df target_name sample:", df[target_name_col].dropna().astype(str).head(5).tolist())

    X_ligand_list, X_prot_list, mols_list, valid_indices = [], [], [], []
    missing_embed = 0
    bad_smiles = 0

    for j, (i, row) in enumerate(df.iterrows(), start=1):
        smi = row.get("smiles", None)
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            bad_smiles += 1
            continue

        # Choose best key: prefer chembl id if present, else name
        key_candidates = []
        if target_id_col in df.columns and row.get(target_id_col, None) is not None:
            key_candidates.append(_norm_key(row[target_id_col]))
        if target_name_col in df.columns and row.get(target_name_col, None) is not None:
            key_candidates.append(_norm_key(row[target_name_col]))

        prot_embed = None
        used_key = None
        for kc in key_candidates:
            if kc in norm_embed_map:
                prot_embed = norm_embed_map[kc]
                used_key = kc
                break

        if prot_embed is None:
            missing_embed += 1
            continue

        try:
            fp = compute_morgan_fingerprint(mol, radius=radius, n_bits=n_bits)
            desc = compute_descriptors(mol)
            lig_feat = np.concatenate([fp.astype(np.float32), desc], axis=0)

            prot_feat = np.concatenate([lig_feat, prot_embed.astype(np.float32)], axis=0)

            X_ligand_list.append(lig_feat)
            X_prot_list.append(prot_feat)
            mols_list.append(mol)
            valid_indices.append(i)
        except Exception:
            # Keep going; very rare
            continue

        if verbose_every and j % verbose_every == 0:
            print(f"  Processed {j} rows... kept {len(valid_indices)}")

    X_ligand = np.asarray(X_ligand_list, dtype=np.float32)
    X_prot = np.asarray(X_prot_list, dtype=np.float32)

    print("\n✓ Feature generation summary")
    print(f"  Input rows: {len(df)}")
    print(f"  Kept rows:  {len(valid_indices)}")
    print(f"  Bad SMILES: {bad_smiles}")
    print(f"  Missing protein embedding: {missing_embed}")
    print(f"  Ligand-only shape:         {X_ligand.shape}")
    print(f"  Protein-conditioned shape: {X_prot.shape}")

    return X_ligand, X_prot, mols_list, valid_indices
