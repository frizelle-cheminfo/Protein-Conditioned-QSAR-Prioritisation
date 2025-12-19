"""
Tests for feature engineering.
"""

import pytest
import numpy as np
from rdkit import Chem

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feature_engineering import (
    compute_morgan_fingerprint,
    compute_descriptors
)


def test_compute_morgan_fingerprint():
    """Test Morgan fingerprint computation."""
    # Aspirin
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(smiles)

    fp = compute_morgan_fingerprint(mol, radius=2, n_bits=2048)

    assert isinstance(fp, np.ndarray)
    assert fp.shape == (2048,)
    assert fp.dtype == np.int8
    assert np.all((fp == 0) | (fp == 1))  # Binary


def test_compute_descriptors():
    """Test physicochemical descriptor computation."""
    # Aspirin
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(smiles)

    desc = compute_descriptors(mol)

    assert isinstance(desc, np.ndarray)
    assert len(desc) == 8
    assert desc.dtype == np.float32

    # Check reasonable ranges
    mw = desc[0]
    assert 100 < mw < 300  # Aspirin MW ~ 180

    logp = desc[1]
    assert -2 < logp < 5


def test_invalid_smiles():
    """Test handling of invalid SMILES."""
    smiles = "INVALID_SMILES"
    mol = Chem.MolFromSmiles(smiles)

    assert mol is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
