"""
Protein embedding generation.

Supports ESM2 transformer models and fallback amino acid composition encodings.
"""

import numpy as np
import pickle
import os


class ESM2Embedder:
    """ESM2-based protein embedder using frozen transformer."""

    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        """
        Initialise ESM2 embedder.

        Args:
            model_name (str): HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.tokeniser = None

    def load_model(self):
        """Load ESM2 model and tokeniser."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            self.tokeniser = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            print(f"✓ Loaded ESM2 model: {self.model_name}")
            return True
        except Exception as e:
            print(f"✗ Failed to load ESM2: {e}")
            return False

    def embed_sequences(self, sequences_dict):
        """
        Compute embeddings for protein sequences.

        Args:
            sequences_dict (dict): Mapping of identifier -> protein sequence

        Returns:
            dict: Mapping of identifier -> embedding vector
        """
        if self.model is None:
            if not self.load_model():
                return None

        import torch

        embeddings = {}

        with torch.no_grad():
            for identifier, seq in sequences_dict.items():
                # Truncate long sequences
                if len(seq) > 1000:
                    seq = seq[:1000]

                inputs = self.tokeniser(seq, return_tensors="pt", truncation=True, max_length=1024)
                outputs = self.model(**inputs)

                # Use mean of sequence representations
                embed = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings[identifier] = embed

                print(f"  {identifier}: {embed.shape}")

        return embeddings


class FallbackEmbedder:
    """Fallback embedder using amino acid composition and dipeptides."""

    def __init__(self):
        """Initialise fallback embedder."""
        self.aa_list = list('ACDEFGHIKLMNPQRSTVWY')

    def embed_sequences(self, sequences_dict):
        """
        Compute composition-based embeddings.

        Args:
            sequences_dict (dict): Mapping of identifier -> protein sequence

        Returns:
            dict: Mapping of identifier -> embedding vector
        """
        embeddings = {}

        for identifier, seq in sequences_dict.items():
            # AA composition
            aa_counts = np.array([seq.count(aa) / len(seq) for aa in self.aa_list])

            # Dipeptide counts (sample, not all)
            dipeptides = [aa1+aa2 for aa1 in 'ACDEFGHIKLMNPQRSTVWY' for aa2 in 'ACDE']
            dipep_counts = np.array([seq.count(dp) / max(1, len(seq)-1) for dp in dipeptides[:80]])

            embed = np.concatenate([aa_counts, dipep_counts])
            embeddings[identifier] = embed

            print(f"  {identifier}: {embed.shape}")

        return embeddings


def fetch_protein_sequences(target_infos):
    """
    Fetch protein sequences from ChEMBL.

    Args:
        target_infos (list): List of dicts with 'chembl_id' and 'name' keys

    Returns:
        dict: Mapping of chembl_id -> protein sequence
    """
    sequences = {}

    try:
        from chembl_webresource_client.new_client import new_client
        target_client = new_client.target

        for target_info in target_infos:
            chembl_id = target_info['chembl_id']
            name = target_info['name']

            try:
                target_data = target_client.get(chembl_id)

                # Try to get sequence from components
                if 'target_components' in target_data and len(target_data['target_components']) > 0:
                    seq = target_data['target_components'][0].get('target_component_sequences', [{}])[0].get('sequence')
                    if seq:
                        sequences[chembl_id] = seq
                        print(f"  {name}: {len(seq)} residues")
                    else:
                        print(f"  {name}: No sequence found")
                else:
                    print(f"  {name}: No components")
            except:
                print(f"  {name}: Fetch failed")
    except:
        print("  ChEMBL API unavailable")

    return sequences


def compute_protein_embeddings(target_infos, cache_path=None, use_esm2=True):
    """
    Compute protein embeddings with caching.

    Args:
        target_infos (list): List of target info dicts
        cache_path (str): Path to cache file
        use_esm2 (bool): Try ESM2 first, fallback if failed

    Returns:
        tuple: (embeddings_dict, embed_type)
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings, 'cached'

    # Fetch sequences
    print("Fetching protein sequences...")
    sequences = fetch_protein_sequences(target_infos)

    if len(sequences) == 0:
        print("\n⚠️ No sequences fetched, using target name hashing as last resort")
        # Create dummy sequences for hashing
        sequences = {t['chembl_id']: t['name']*50 for t in target_infos}

    # Try ESM2
    embeddings = None
    embed_type = None

    if use_esm2:
        print("\nAttempting ESM2 embeddings...")
        embedder = ESM2Embedder()
        embeddings = embedder.embed_sequences(sequences)
        if embeddings:
            embed_type = 'esm2'

    # Fallback
    if embeddings is None:
        print("\nUsing fallback embeddings (AA composition)...")
        embedder = FallbackEmbedder()
        embeddings = embedder.embed_sequences(sequences)
        embed_type = 'fallback'

    # Cache
    if cache_path and embeddings:
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"✓ Cached embeddings to {cache_path}")

    return embeddings, embed_type
