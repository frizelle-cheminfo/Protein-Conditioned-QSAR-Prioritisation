"""
PDF reporting and visualisation.

Generates comprehensive PDF reports with plots and tables.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


def create_comprehensive_report(
    output_path,
    targets,
    primary_target,
    model_type,
    n_ensemble,
    df_within,
    df_loto,
    df_enrichment,
    df_prioritised,
    plots_dir
):
    """
    Generate comprehensive PDF report.

    Args:
        output_path (str): Path to save PDF
        targets (list): List of target names
        primary_target (str): Primary target name
        model_type (str): Model type (rf_ensemble, xgb_ensemble)
        n_ensemble (int): Number of models in ensemble
        df_within (pd.DataFrame): Within-target benchmark results
        df_loto (pd.DataFrame): LOTO benchmark results
        df_enrichment (pd.DataFrame): Enrichment results
        df_prioritised (pd.DataFrame): Prioritised compounds
        plots_dir (str): Directory containing plots
    """
    print("\nGenerating PDF report...")

    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.7, 'Protein-Conditioned QSAR Report', ha='center', fontsize=22, weight='bold')
        fig.text(0.5, 0.6, f'Targets: {", ".join(targets)}', ha='center', fontsize=14)
        fig.text(0.5, 0.5, f'Primary: {primary_target}', ha='center', fontsize=14)
        fig.text(0.5, 0.4, f'Model: {model_type} (N={n_ensemble})', ha='center', fontsize=12)
        fig.text(0.5, 0.3, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ha='center', fontsize=10)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Benchmark summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.95, 'Benchmark Summary', ha='center', fontsize=18, weight='bold')

        summary_text = f"""
Within-Target Scaffold Split (Spearman):
{df_within.groupby('features')['spearman'].mean().to_string()}

Leave-One-Target-Out (Spearman):
{df_loto.groupby('features')['spearman'].mean().to_string()}

Enrichment@10% (mean):
{df_enrichment.groupby('features')['enrichment@10%'].mean().to_string()}
        """

        fig.text(0.1, 0.8, summary_text, fontsize=10, verticalalignment='top', family='monospace')
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Add plots
        plot_files = [
            os.path.join(plots_dir, 'benchmark_comparison.png'),
            os.path.join(plots_dir, 'enrichment_comparison.png')
        ]

        for plot_file in plot_files:
            if os.path.exists(plot_file):
                img = plt.imread(plot_file)
                fig = plt.figure(figsize=(11, 8.5))
                plt.imshow(img)
                plt.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        # Top 20 table
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.95, f'Top 20 Compounds ({primary_target})', ha='center', fontsize=16, weight='bold')

        top_20_data = df_prioritised.head(20)[[
            'pred_pActivity', 'selectivity', 'makeability', 'combined_score'
        ]].round(3)

        ax = fig.add_subplot(111)
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=top_20_data.values,
            colLabels=top_20_data.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"✓ Report saved to {output_path}")


def create_benchmark_plots(df_within, df_loto, df_enrichment, output_dir):
    """
    Create benchmark comparison plots.

    Args:
        df_within (pd.DataFrame): Within-target results
        df_loto (pd.DataFrame): LOTO results
        df_enrichment (pd.DataFrame): Enrichment results
        output_dir (str): Directory to save plots
    """
    # Within-target comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Spearman
    ax = axes[0]
    pivot = df_within.pivot(index='target', columns='features', values='spearman')
    x = np.arange(len(pivot))
    width = 0.35
    ax.bar(x - width/2, pivot['ligand_only'], width, label='Ligand-only', alpha=0.8)
    ax.bar(x + width/2, pivot['protein_conditioned'], width, label='Protein-conditioned', alpha=0.8)
    ax.set_xlabel('Target', fontsize=12)
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('Within-Target Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # LOTO
    ax = axes[1]
    pivot_loto = df_loto.pivot(index='test_target', columns='features', values='spearman')
    x = np.arange(len(pivot_loto))
    ax.bar(x - width/2, pivot_loto['ligand_only'], width, label='Ligand-only', alpha=0.8)
    ax.bar(x + width/2, pivot_loto['protein_conditioned'], width, label='Protein-conditioned', alpha=0.8)
    ax.set_xlabel('Test Target', fontsize=12)
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('Leave-One-Target-Out', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_loto.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Enrichment
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_enrich = df_enrichment.pivot(index='target', columns='features', values='enrichment@10%')
    x = np.arange(len(pivot_enrich))
    ax.bar(x - width/2, pivot_enrich['ligand_only'], width, label='Ligand-only', alpha=0.8)
    ax.bar(x + width/2, pivot_enrich['protein_conditioned'], width, label='Protein-conditioned', alpha=0.8)
    ax.axhline(1.0, color='red', linestyle='--', label='Baseline', alpha=0.5)
    ax.set_xlabel('Target', fontsize=12)
    ax.set_ylabel('Enrichment@10% (fold)', fontsize=12)
    ax.set_title('Active Compound Enrichment', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_enrich.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enrichment_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("\n✓ Benchmark plots saved")
