"""Machine learning models and evaluation."""

from .ensemble import RFEnsemble, XGBEnsemble
from .evaluation import evaluate_model, compute_enrichment
from .protein_embeddings import ESM2Embedder, FallbackEmbedder

__all__ = [
    'RFEnsemble',
    'XGBEnsemble',
    'evaluate_model',
    'compute_enrichment',
    'ESM2Embedder',
    'FallbackEmbedder'
]
