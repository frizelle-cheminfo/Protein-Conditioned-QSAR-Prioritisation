"""
Compound prioritisation with multi-objective scoring.

Combines potency, selectivity, makeability, and uncertainty into a single score.
"""

import numpy as np
import pandas as pd
from rdkit import Chem

from ..data.feature_engineering import compute_morgan_fingerprint, compute_descriptors
from ..models.ensemble import predict_with_uncertainty
from .makeability import compute_makeability_score


def prioritise_with_selectivity(
    smiles_list,
    primary_models,
    offtarget_models,
    target_embed,
    offtarget_embeds,
    lambda_uncertainty=0.5,
    weight_potency=0.5,
    weight_selectivity=0.2,
    weight_makeability=0.2,
    weight_uncertainty=0.1
):
    """
    Prioritise compounds with selectivity scoring.

    Args:
        smiles_list (list): List of SMILES strings to score
        primary_models (list): Ensemble models for primary target
        offtarget_models (dict): Dict of {target_name: models} for off-targets
        target_embed (np.ndarray): Protein embedding for primary target
        offtarget_embeds (dict): Dict of {target_name: embedding} for off-targets
        lambda_uncertainty (float): Uncertainty penalty coefficient
        weight_potency (float): Weight for potency in combined score
        weight_selectivity (float): Weight for selectivity in combined score
        weight_makeability (float): Weight for makeability in combined score
        weight_uncertainty (float): Weight for uncertainty in combined score

    Returns:
        pd.DataFrame: Prioritised compounds sorted by combined score
            Columns:
            - smiles
            - pred_pActivity: Predicted potency
            - pred_uncertainty: Prediction uncertainty
            - pred_conservative: Conservative score (mean - λ*std)
            - pred_offmax: Maximum predicted off-target activity
            - selectivity: On-target - max(off-targets)
            - makeability: Synthetic accessibility score
            - combined_score: Weighted combination
    """
    results = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        try:
            # Ligand features
            fp = compute_morgan_fingerprint(mol)
            desc = compute_descriptors(mol)
            lig_feat = np.concatenate([fp, desc])

            # Primary target prediction
            prot_feat_primary = np.concatenate([lig_feat, target_embed]).reshape(1, -1)
            pred_primary, std_primary = predict_with_uncertainty(primary_models, prot_feat_primary)
            pred_primary = pred_primary[0]
            std_primary = std_primary[0]

            # Off-target predictions
            pred_offmax = 0.0
            if offtarget_models:
                off_preds = []
                for off_name, off_models in offtarget_models.items():
                    off_embed = offtarget_embeds[off_name]
                    prot_feat_off = np.concatenate([lig_feat, off_embed]).reshape(1, -1)
                    pred_off, _ = predict_with_uncertainty(off_models, prot_feat_off)
                    off_preds.append(pred_off[0])
                pred_offmax = max(off_preds) if off_preds else 0.0

            selectivity = pred_primary - pred_offmax

            # Conservative score
            pred_conservative = pred_primary - lambda_uncertainty * std_primary

            # Makeability
            makeability = compute_makeability_score(mol)

            # Combined score
            norm_potency = (pred_primary - 3.0) / 8.0  # Normalise to [0, 1]
            norm_selectivity = (selectivity + 2.0) / 4.0  # Assume range [-2, 2]
            norm_uncertainty = 1.0 - (std_primary / 2.0)  # Lower uncertainty = better

            combined_score = (
                weight_potency * norm_potency +
                weight_selectivity * norm_selectivity +
                weight_makeability * makeability +
                weight_uncertainty * norm_uncertainty
            )

            results.append({
                'smiles': smiles,
                'pred_pActivity': pred_primary,
                'pred_uncertainty': std_primary,
                'pred_conservative': pred_conservative,
                'pred_offmax': pred_offmax,
                'selectivity': selectivity,
                'makeability': makeability,
                'combined_score': combined_score
            })
        except:
            continue

    df = pd.DataFrame(results)
    df = df.sort_values('combined_score', ascending=False).reset_index(drop=True)

    return df
