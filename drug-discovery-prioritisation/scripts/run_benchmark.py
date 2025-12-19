#!/usr/bin/env python3
"""
Run comprehensive benchmarking pipeline.

Reproduces benchmark results from config file.
"""

import argparse
import yaml
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.chembl_downloader import download_chembl_target
from src.data.preprocessing import clean_and_compute_pActivity
from src.data.feature_engineering import generate_features_multitarget
from src.models.protein_embeddings import compute_protein_embeddings
from src.models.ensemble import train_rf_ensemble, train_xgb_ensemble, predict_with_uncertainty
from src.models.evaluation import evaluate_model, compute_enrichment
from src.utils.splits import scaffold_split_within_target, leave_one_target_out_split
from src.utils.reporting import create_comprehensive_report, create_benchmark_plots


def main():
    parser = argparse.ArgumentParser(description='Run benchmarking pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"{'='*60}")
    print("DRUG DISCOVERY PRIORITISATION - BENCHMARK")
    print(f"{'='*60}")
    print(f"Config: {args.config}")

    # Create output directory
    output_dir = config.get('output_dir', './outputs')
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    Path(f'{output_dir}/benchmarks').mkdir(exist_ok=True)
    Path(f'{output_dir}/plots').mkdir(exist_ok=True)

    # Download data
    print(f"\n{'='*60}")
    print("DATA ACQUISITION")
    print(f"{'='*60}")

    dfs = []
    for target_info in config['targets']:
        df = download_chembl_target(
            target_info,
            config.get('activity_type', 'IC50'),
            config.get('max_compounds', 1000)
        )
        if df is not None:
            dfs.append(df)

    raw_data = pd.concat(dfs, ignore_index=True)
    df_clean = clean_and_compute_pActivity(raw_data)

    # Filter targets
    min_samples = config.get('min_samples_per_target', 200)
    target_counts = df_clean.groupby('target_name').size()
    valid_targets = target_counts[target_counts >= min_samples].index.tolist()
    df_clean = df_clean[df_clean['target_name'].isin(valid_targets)]

    print(f"\nValid targets: {valid_targets}")

    # Compute protein embeddings
    print(f"\n{'='*60}")
    print("PROTEIN EMBEDDINGS")
    print(f"{'='*60}")

    cache_path = f'{output_dir}/cache_protein_embeds.pkl'
    protein_embeds, embed_type = compute_protein_embeddings(
        config['targets'],
        cache_path=cache_path
    )

    # Create target_name -> embedding map
    target_embed_map = {}
    for target_info in config['targets']:
        chembl_id = target_info['chembl_id']
        name = target_info['name']
        if chembl_id in protein_embeds:
            target_embed_map[name] = protein_embeds[chembl_id]

    # Generate features
    print(f"\n{'='*60}")
    print("FEATURE GENERATION")
    print(f"{'='*60}")

    X_ligand, X_protein, mols, valid_indices = generate_features_multitarget(
        df_clean, target_embed_map
    )

    df_features = df_clean.loc[valid_indices].reset_index(drop=True)
    y = df_features["pActivity"].values

    # Generate splits
    print(f"\n{'='*60}")
    print("DATA SPLITS")
    print(f"{'='*60}")

    scaffold_splits = scaffold_split_within_target(df_features, random_state=config.get('random_seed', 42))
    loto_splits = leave_one_target_out_split(df_features)

    # Benchmark: Within-target
    print(f"\n{'='*60}")
    print("BENCHMARK: WITHIN-TARGET SCAFFOLD SPLIT")
    print(f"{'='*60}")

    model_type = config.get('model_type', 'rf_ensemble')
    n_ensemble = config.get('n_ensemble', 5)
    random_seed = config.get('random_seed', 42)

    within_target_results = []

    for target_name, (train_idx, val_idx, test_idx) in scaffold_splits.items():
        print(f"\n{target_name}:")

        # Ligand-only
        if model_type == "rf_ensemble":
            models_lig = train_rf_ensemble(X_ligand[train_idx], y[train_idx], n_ensemble, random_seed)
        else:
            models_lig = train_xgb_ensemble(X_ligand[train_idx], y[train_idx], n_ensemble, random_seed)

        y_pred_lig, y_std_lig = predict_with_uncertainty(models_lig, X_ligand[test_idx])
        metrics_lig = evaluate_model(y[test_idx], y_pred_lig, y_std_lig)

        within_target_results.append({
            'target': target_name,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'features': 'ligand_only',
            **metrics_lig
        })

        # Protein-conditioned
        if model_type == "rf_ensemble":
            models_prot = train_rf_ensemble(X_protein[train_idx], y[train_idx], n_ensemble, random_seed)
        else:
            models_prot = train_xgb_ensemble(X_protein[train_idx], y[train_idx], n_ensemble, random_seed)

        y_pred_prot, y_std_prot = predict_with_uncertainty(models_prot, X_protein[test_idx])
        metrics_prot = evaluate_model(y[test_idx], y_pred_prot, y_std_prot)

        within_target_results.append({
            'target': target_name,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'features': 'protein_conditioned',
            **metrics_prot
        })

        print(f"  Ligand-only:    Spearman={metrics_lig['spearman']:.3f}")
        print(f"  Protein-cond:   Spearman={metrics_prot['spearman']:.3f}")

    df_within = pd.DataFrame(within_target_results)
    df_within.to_csv(f'{output_dir}/benchmarks/within_target_scaffold.csv', index=False)

    # Benchmark: LOTO
    print(f"\n{'='*60}")
    print("BENCHMARK: LEAVE-ONE-TARGET-OUT")
    print(f"{'='*60}")

    loto_results = []

    for train_idx, test_idx, test_target in loto_splits:
        print(f"\nTest target: {test_target}")

        # Ligand-only
        if model_type == "rf_ensemble":
            models_lig = train_rf_ensemble(X_ligand[train_idx], y[train_idx], n_ensemble, random_seed)
        else:
            models_lig = train_xgb_ensemble(X_ligand[train_idx], y[train_idx], n_ensemble, random_seed)

        y_pred_lig, y_std_lig = predict_with_uncertainty(models_lig, X_ligand[test_idx])
        metrics_lig = evaluate_model(y[test_idx], y_pred_lig, y_std_lig)

        loto_results.append({
            'test_target': test_target,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'features': 'ligand_only',
            **metrics_lig
        })

        # Protein-conditioned
        if model_type == "rf_ensemble":
            models_prot = train_rf_ensemble(X_protein[train_idx], y[train_idx], n_ensemble, random_seed)
        else:
            models_prot = train_xgb_ensemble(X_protein[train_idx], y[train_idx], n_ensemble, random_seed)

        y_pred_prot, y_std_prot = predict_with_uncertainty(models_prot, X_protein[test_idx])
        metrics_prot = evaluate_model(y[test_idx], y_pred_prot, y_std_prot)

        loto_results.append({
            'test_target': test_target,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'features': 'protein_conditioned',
            **metrics_prot
        })

        print(f"  Ligand-only:    Spearman={metrics_lig['spearman']:.3f}")
        print(f"  Protein-cond:   Spearman={metrics_prot['spearman']:.3f}")

    df_loto = pd.DataFrame(loto_results)
    df_loto.to_csv(f'{output_dir}/benchmarks/leave_one_target_out.csv', index=False)

    # Benchmark: Enrichment
    print(f"\n{'='*60}")
    print("BENCHMARK: ENRICHMENT ANALYSIS")
    print(f"{'='*60}")

    enrichment_results = []
    activity_threshold = config.get('activity_threshold', 7.0)
    enrichment_fractions = config.get('enrichment_fractions', [0.05, 0.10])

    for target_name, (train_idx, val_idx, test_idx) in scaffold_splits.items():
        print(f"\n{target_name}:")

        # Train models
        if model_type == "rf_ensemble":
            models_lig = train_rf_ensemble(X_ligand[train_idx], y[train_idx], n_ensemble, random_seed)
            models_prot = train_rf_ensemble(X_protein[train_idx], y[train_idx], n_ensemble, random_seed)
        else:
            models_lig = train_xgb_ensemble(X_ligand[train_idx], y[train_idx], n_ensemble, random_seed)
            models_prot = train_xgb_ensemble(X_protein[train_idx], y[train_idx], n_ensemble, random_seed)

        y_test = y[test_idx]

        # Ligand-only
        y_pred_lig, y_std_lig = predict_with_uncertainty(models_lig, X_ligand[test_idx])
        enrich_lig = compute_enrichment(y_test, y_pred_lig, activity_threshold, enrichment_fractions)

        enrichment_results.append({
            'target': target_name,
            'features': 'ligand_only',
            'n_test': len(test_idx),
            'n_actives': (y_test >= activity_threshold).sum(),
            **enrich_lig
        })

        # Protein-conditioned
        y_pred_prot, y_std_prot = predict_with_uncertainty(models_prot, X_protein[test_idx])
        enrich_prot = compute_enrichment(y_test, y_pred_prot, activity_threshold, enrichment_fractions)

        enrichment_results.append({
            'target': target_name,
            'features': 'protein_conditioned',
            'n_test': len(test_idx),
            'n_actives': (y_test >= activity_threshold).sum(),
            **enrich_prot
        })

        print(f"  Ligand-only:    Enrichment@10%={enrich_lig['enrichment@10%']:.2f}x")
        print(f"  Protein-cond:   Enrichment@10%={enrich_prot['enrichment@10%']:.2f}x")

    df_enrichment = pd.DataFrame(enrichment_results)
    df_enrichment.to_csv(f'{output_dir}/benchmarks/enrichment.csv', index=False)

    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")

    create_benchmark_plots(df_within, df_loto, df_enrichment, f'{output_dir}/plots')

    # Generate report
    # Note: For full report, would need prioritised compounds
    # This is a simplified version
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"  - benchmarks/within_target_scaffold.csv")
    print(f"  - benchmarks/leave_one_target_out.csv")
    print(f"  - benchmarks/enrichment.csv")
    print(f"  - plots/")


if __name__ == '__main__':
    main()
