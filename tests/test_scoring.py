"""
Tests for scoring functions.
"""

import pytest
import numpy as np
from rdkit import Chem

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring.makeability import (
    compute_sa_score_improved,
    compute_makeability_score
)


def test_sa_score_simple_molecule():
    """Test SA score for simple molecule."""
    # Benzene (very simple)
    smiles = "c1ccccc1"
    mol = Chem.MolFromSmiles(smiles)

    sa_score = compute_sa_score_improved(mol)

    # Benzene should have low complexity
    assert 1.0 <= sa_score <= 3.0


def test_sa_score_complex_molecule():
    """Test SA score for complex molecule."""
    # Taxol (very complex)
    smiles = "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
    mol = Chem.MolFromSmiles(smiles)

    sa_score = compute_sa_score_improved(mol)

    # Taxol should have high complexity
    assert sa_score >= 5.0


def test_makeability_score_range():
    """Test that makeability score is in valid range."""
    test_smiles = [
        "c1ccccc1",  # Benzene
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CCN(CC)C(=O)Nc1ccc2ncnc(Nc3cccc(Br)c3)c2c1"  # Drug-like
    ]

    for smiles in test_smiles:
        mol = Chem.MolFromSmiles(smiles)
        score = compute_makeability_score(mol)

        assert 0.0 <= score <= 1.0


def test_makeability_mw_penalty():
    """Test molecular weight penalty."""
    # Large molecule (MW > 550)
    large_smiles = "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
    large_mol = Chem.MolFromSmiles(large_smiles)

    # Small molecule (MW < 550)
    small_smiles = "c1ccccc1"
    small_mol = Chem.MolFromSmiles(small_smiles)

    score_large = compute_makeability_score(large_mol, mw_threshold=550)
    score_small = compute_makeability_score(small_mol, mw_threshold=550)

    # Smaller molecule should have better makeability
    assert score_small > score_large


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
