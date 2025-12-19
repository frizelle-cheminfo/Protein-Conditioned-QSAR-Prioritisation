"""
Ensemble models for uncertainty-aware prediction.

Implements Random Forest and XGBoost ensembles with uncertainty quantification.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


class RFEnsemble:
    """Random Forest ensemble with uncertainty estimation."""

    def __init__(self, n_ensemble=5, n_estimators=100, max_depth=20,
                 min_samples_split=5, random_state=42, n_jobs=-1):
        """
        Initialise RF ensemble.

        Args:
            n_ensemble (int): Number of models in ensemble
            n_estimators (int): Number of trees per model
            max_depth (int): Maximum tree depth
            min_samples_split (int): Minimum samples to split node
            random_state (int): Random seed
            n_jobs (int): Number of parallel jobs
        """
        self.n_ensemble = n_ensemble
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = []

    def fit(self, X_train, y_train):
        """
        Train ensemble models.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        self.models = []
        for i in range(self.n_ensemble):
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_jobs=self.n_jobs,
                random_state=self.random_state + i
            )
            model.fit(X_train, y_train)
            self.models.append(model)

    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty estimation.

        Args:
            X (np.ndarray): Features to predict

        Returns:
            tuple: (predictions, uncertainties)
                - predictions: Mean predictions across ensemble
                - uncertainties: Standard deviation across ensemble
        """
        preds = np.array([m.predict(X) for m in self.models])
        return preds.mean(axis=0), preds.std(axis=0)


class XGBEnsemble:
    """XGBoost ensemble with uncertainty estimation."""

    def __init__(self, n_ensemble=5, n_estimators=100, max_depth=6,
                 learning_rate=0.1, random_state=42):
        """
        Initialise XGB ensemble.

        Args:
            n_ensemble (int): Number of models in ensemble
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            learning_rate (float): Learning rate
            random_state (int): Random seed
        """
        self.n_ensemble = n_ensemble
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.models = []

    def fit(self, X_train, y_train):
        """
        Train ensemble models.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        self.models = []
        for i in range(self.n_ensemble):
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state + i
            )
            model.fit(X_train, y_train, verbose=False)
            self.models.append(model)

    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty estimation.

        Args:
            X (np.ndarray): Features to predict

        Returns:
            tuple: (predictions, uncertainties)
                - predictions: Mean predictions across ensemble
                - uncertainties: Standard deviation across ensemble
        """
        preds = np.array([m.predict(X) for m in self.models])
        return preds.mean(axis=0), preds.std(axis=0)


def train_rf_ensemble(X_train, y_train, n_ensemble=5, random_state=42):
    """
    Convenience function to train RF ensemble.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        n_ensemble (int): Number of models
        random_state (int): Random seed

    Returns:
        list: List of trained RandomForest models
    """
    models = []
    for i in range(n_ensemble):
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            n_jobs=-1,
            random_state=random_state + i
        )
        model.fit(X_train, y_train)
        models.append(model)
    return models


def train_xgb_ensemble(X_train, y_train, n_ensemble=5, random_state=42):
    """
    Convenience function to train XGB ensemble.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        n_ensemble (int): Number of models
        random_state (int): Random seed

    Returns:
        list: List of trained XGBoost models
    """
    models = []
    for i in range(n_ensemble):
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state + i
        )
        model.fit(X_train, y_train, verbose=False)
        models.append(model)
    return models


def predict_with_uncertainty(models, X):
    """
    Predict using model ensemble with uncertainty.

    Args:
        models (list): List of trained models
        X (np.ndarray): Features to predict

    Returns:
        tuple: (predictions, uncertainties)
    """
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0), preds.std(axis=0)
