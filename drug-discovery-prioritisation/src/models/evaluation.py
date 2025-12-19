"""
Model evaluation metrics.

Includes regression metrics, enrichment analysis, and uncertainty calibration.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr, pearsonr


def evaluate_model(y_true, y_pred, y_std):
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        y_std (np.ndarray): Prediction uncertainties

    Returns:
        dict: Dictionary of evaluation metrics
            - r2: R² score
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - spearman: Spearman rank correlation
            - pearson: Pearson correlation
            - unc_corr: Uncertainty calibration (correlation with absolute error)
            - mean_std: Mean prediction uncertainty
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    spearman = spearmanr(y_true, y_pred)[0]
    pearson = pearsonr(y_true, y_pred)[0]

    # Uncertainty calibration
    abs_errors = np.abs(y_true - y_pred)
    unc_corr = np.corrcoef(y_std, abs_errors)[0, 1] if len(y_std) > 1 else 0.0

    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'spearman': spearman,
        'pearson': pearson,
        'unc_corr': unc_corr,
        'mean_std': y_std.mean()
    }


def compute_enrichment(y_true, y_pred, threshold, fractions):
    """
    Compute enrichment at top fractions.

    Measures how well the model ranks active compounds (above threshold)
    in the top-ranked predictions.

    Args:
        y_true (np.ndarray): True activity values
        y_pred (np.ndarray): Predicted activity values
        threshold (float): Activity threshold for "active" classification
        fractions (list): List of top fractions to evaluate (e.g., [0.05, 0.10])

    Returns:
        dict: Enrichment and recall metrics at each fraction
            - enrichment@X%: Fold-enrichment over random (1.0 = random)
            - recall@X%: Fraction of actives recovered in top X%
    """
    actives = (y_true >= threshold).astype(int)
    baseline_rate = actives.mean()

    if baseline_rate == 0:
        return {f'enrichment@{int(f*100)}%': 0.0 for f in fractions}

    # Rank by prediction
    ranked_idx = np.argsort(y_pred)[::-1]

    results = {}
    for frac in fractions:
        k = max(1, int(len(y_true) * frac))
        top_k_actives = actives[ranked_idx[:k]].sum()
        hit_rate = top_k_actives / k
        enrichment = hit_rate / baseline_rate
        results[f'enrichment@{int(frac*100)}%'] = enrichment
        results[f'recall@{int(frac*100)}%'] = top_k_actives / actives.sum() if actives.sum() > 0 else 0.0

    return results
