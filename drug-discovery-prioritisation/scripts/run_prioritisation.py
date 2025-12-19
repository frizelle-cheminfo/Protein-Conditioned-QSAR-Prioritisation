#!/usr/bin/env python3
"""
Score and prioritise a compound library.

Loads pre-trained models and scores new compounds.
"""

import argparse
import pickle
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring.prioritisation import prioritise_with_selectivity


def main():
    parser = argparse.ArgumentParser(description='Prioritise compound library')
    parser.add_argument('--compounds', type=str, required=True, help='CSV with SMILES column')
    parser.add_argument('--models-dir', type=str, required=True, help='Directory with trained models')
    parser.add_argument('--primary-target', type=str, required=True, help='Primary target name')
    parser.add_argument('--off-targets', type=str, nargs='+', help='Off-target names')
    parser.add_argument('--output', type=str, default='prioritised_compounds.csv', help='Output CSV path')
    parser.add_argument('--top-n', type=int, help='Also save top N to separate file')
    args = parser.parse_args()

    print(f"{'='*60}")
    print("COMPOUND PRIORITISATION")
    print(f"{'='*60}")

    # Load compounds
    print(f"\nLoading compounds from {args.compounds}...")
    df_compounds = pd.read_csv(args.compounds)

    if 'smiles' not in df_compounds.columns and 'SMILES' not in df_compounds.columns:
        raise ValueError("CSV must have 'smiles' or 'SMILES' column")

    if 'SMILES' in df_compounds.columns:
        df_compounds = df_compounds.rename(columns={'SMILES': 'smiles'})

    smiles_list = df_compounds['smiles'].dropna().tolist()
    print(f"  Found {len(smiles_list)} SMILES")

    # Load models
    print(f"\nLoading models from {args.models_dir}...")

    models_path = Path(args.models_dir)

    # Load primary target models
    primary_model_file = models_path / f"{args.primary_target}_models.pkl"
    with open(primary_model_file, 'rb') as f:
        primary_models = pickle.load(f)
    print(f"  Primary ({args.primary_target}): {len(primary_models)} models")

    # Load primary target embedding
    primary_embed_file = models_path / f"{args.primary_target}_embedding.pkl"
    with open(primary_embed_file, 'rb') as f:
        primary_embed = pickle.load(f)

    # Load off-target models
    offtarget_models = {}
    offtarget_embeds = {}

    if args.off_targets:
        for off_target in args.off_targets:
            off_model_file = models_path / f"{off_target}_models.pkl"
            off_embed_file = models_path / f"{off_target}_embedding.pkl"

            if off_model_file.exists() and off_embed_file.exists():
                with open(off_model_file, 'rb') as f:
                    offtarget_models[off_target] = pickle.load(f)
                with open(off_embed_file, 'rb') as f:
                    offtarget_embeds[off_target] = pickle.load(f)
                print(f"  Off-target ({off_target}): {len(offtarget_models[off_target])} models")
            else:
                print(f"  Warning: Models for {off_target} not found, skipping")

    # Prioritise
    print(f"\nPrioritising {len(smiles_list)} compounds...")

    df_prioritised = prioritise_with_selectivity(
        smiles_list,
        primary_models,
        offtarget_models,
        primary_embed,
        offtarget_embeds
    )

    print(f"\n✓ Prioritised {len(df_prioritised)} compounds")

    # Save results
    df_prioritised.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    if args.top_n:
        top_n_file = args.output.replace('.csv', f'_top_{args.top_n}.csv')
        df_prioritised.head(args.top_n).to_csv(top_n_file, index=False)
        print(f"Top {args.top_n} saved to: {top_n_file}")

    # Print top 10
    print(f"\nTop 10 compounds:")
    print(df_prioritised[['smiles', 'pred_pActivity', 'selectivity', 'makeability', 'combined_score']].head(10))


if __name__ == '__main__':
    main()
