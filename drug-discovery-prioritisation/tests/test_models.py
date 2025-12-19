"""
Tests for ensemble models.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ensemble import (
    RFEnsemble,
    XGBEnsemble,
    predict_with_uncertainty
)


@pytest.fixture
def synthetic_data():
    """Generate synthetic training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples) * 2 + 5  # pActivity ~ 5 ± 2

    return X, y


def test_rf_ensemble_training(synthetic_data):
    """Test Random Forest ensemble training."""
    X, y = synthetic_data

    ensemble = RFEnsemble(n_ensemble=3, n_estimators=10, random_state=42)
    ensemble.fit(X, y)

    assert len(ensemble.models) == 3


def test_rf_ensemble_prediction(synthetic_data):
    """Test Random Forest ensemble prediction with uncertainty."""
    X, y = synthetic_data

    ensemble = RFEnsemble(n_ensemble=3, n_estimators=10, random_state=42)
    ensemble.fit(X[:80], y[:80])

    preds, stds = ensemble.predict_with_uncertainty(X[80:])

    assert len(preds) == 20
    assert len(stds) == 20
    assert np.all(stds >= 0)  # Uncertainty should be non-negative


def test_xgb_ensemble_training(synthetic_data):
    """Test XGBoost ensemble training."""
    X, y = synthetic_data

    ensemble = XGBEnsemble(n_ensemble=3, n_estimators=10, random_state=42)
    ensemble.fit(X, y)

    assert len(ensemble.models) == 3


def test_uncertainty_quantification(synthetic_data):
    """Test that uncertainty correlates with difficulty."""
    X, y = synthetic_data

    # Add some outliers (harder to predict)
    X_outliers = np.random.randn(10, 50) * 10
    y_outliers = np.random.randn(10) * 5

    ensemble = RFEnsemble(n_ensemble=5, n_estimators=20, random_state=42)
    ensemble.fit(X, y)

    # Normal predictions
    _, std_normal = ensemble.predict_with_uncertainty(X[:20])

    # Outlier predictions
    _, std_outlier = ensemble.predict_with_uncertainty(X_outliers)

    # Outliers should have higher uncertainty on average
    assert std_outlier.mean() > std_normal.mean()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
