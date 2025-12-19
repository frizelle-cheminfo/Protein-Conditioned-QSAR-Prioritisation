"""
Synthetic accessibility and makeability scoring.

Heuristic scoring based on molecular complexity, molecular weight, and structural features.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def compute_sa_score_improved(mol):
    """
    Compute synthetic accessibility score (1-10, higher = more complex).

    Based on molecular complexity heuristics:
    - Ring systems
    - Spiro and bridgehead atoms
    - Stereocentres
    - Heavy atom count

    Args:
        mol: RDKit molecule object

    Returns:
        float: SA score (1-10)
    """
    complexity = 0.0
    ri = mol.GetRingInfo()
    n_rings = len(ri.AtomRings())
    complexity += n_rings * 0.5

    for ring in ri.AtomRings():
        if len(ring) > 8:
            complexity += 2.0

    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    complexity += n_spiro * 1.0 + n_bridge * 0.5

    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    complexity += n_stereo * 0.5

    n_atoms = mol.GetNumHeavyAtoms()
    if n_atoms > 50:
        complexity += (n_atoms - 50) * 0.1

    return 1.0 + min(9.0, complexity)


def compute_makeability_score(mol, mw_threshold=550, rotatable_bonds_threshold=10,
                              ring_count_threshold=4):
    """
    Compute overall makeability score (0-1, higher = more makeable).

    Combines:
    - Synthetic accessibility (inverted)
    - Molecular weight penalty
    - Rotatable bonds penalty
    - Ring count penalty

    Args:
        mol: RDKit molecule object
        mw_threshold (float): Molecular weight threshold
        rotatable_bonds_threshold (int): Rotatable bonds threshold
        ring_count_threshold (int): Ring count threshold

    Returns:
        float: Makeability score (0-1)
    """
    sa_score = compute_sa_score_improved(mol)
    sa_component = (10 - sa_score) / 9.0

    mw = Descriptors.MolWt(mol)
    mw_penalty = max(0, 1.0 - (mw - mw_threshold) / 300) if mw > mw_threshold else 1.0

    n_rot = Lipinski.NumRotatableBonds(mol)
    rot_penalty = max(0, 1.0 - (n_rot - rotatable_bonds_threshold) * 0.1) if n_rot > rotatable_bonds_threshold else 1.0

    n_rings = rdMolDescriptors.CalcNumRings(mol)
    ring_penalty = max(0, 1.0 - (n_rings - ring_count_threshold) * 0.2) if n_rings > ring_count_threshold else 1.0

    return 0.5 * sa_component + 0.2 * mw_penalty + 0.15 * rot_penalty + 0.15 * ring_penalty
