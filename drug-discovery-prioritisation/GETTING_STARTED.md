# Getting Started

Quick guide to using the drug discovery prioritisation system.

## Installation

### Option 1: Conda (Recommended)

```bash
cd drug-discovery-prioritisation
conda env create -f environment.yml
conda activate drug-prioritisation
```

### Option 2: Pip

```bash
cd drug-discovery-prioritisation
pip install -r requirements.txt
```

## Quick Test

Run the test suite to verify installation:

```bash
pytest tests/ -v
```

## Running Your First Benchmark

Use one of the pre-configured case studies:

```bash
python scripts/run_benchmark.py --config configs/case_study_erbb.yaml
```

This will:
1. Download ChEMBL data for EGFR, ALK, JAK2
2. Train protein-conditioned QSAR models
3. Run benchmarks (scaffold split, LOTO, enrichment)
4. Generate plots and CSV results
5. Save outputs to `outputs/case_study_erbb/`

**Expected runtime:** ~15-30 minutes on a laptop CPU

## Exploring the Notebook

The original exploratory notebook is in `notebooks/`:

```bash
jupyter notebook notebooks/Drug_Discovery_Prioritization_MVP_v3.ipynb
```

This notebook contains:
- Full pipeline walkthrough
- Visualisations
- Case study results
- Interactive exploration

## Next Steps

### Create Your Own Config

Copy `configs/default.yaml` and modify:

```yaml
targets:
  - chembl_id: "YOUR_TARGET_ID"
    name: "YOUR_TARGET_NAME"

primary_target: "YOUR_TARGET_NAME"
output_dir: "./outputs/my_experiment"
```

Then run:

```bash
python scripts/run_benchmark.py --config configs/my_experiment.yaml
```

### Score Your Own Compounds

If you have pre-trained models and a library of compounds:

```bash
python scripts/run_prioritisation.py \
  --compounds my_compounds.csv \
  --models-dir models/egfr_ensemble/ \
  --primary-target EGFR \
  --off-targets ALK JAK2 \
  --output prioritised_results.csv \
  --top-n 20
```

Your CSV must have a `smiles` column.

## Project Structure

```
.
├── src/                 # Reusable modules
│   ├── data/           # Data download and preprocessing
│   ├── models/         # Ensemble models and protein embeddings
│   ├── scoring/        # Prioritisation and makeability
│   └── utils/          # Splits, plotting, reporting
│
├── scripts/            # Runnable pipelines
│   ├── run_benchmark.py
│   └── run_prioritisation.py
│
├── configs/            # Experiment configurations
├── notebooks/          # Exploratory notebooks
├── tests/              # Unit tests
└── outputs/            # Results (gitignored)
```

## Common Issues

### 1. ChEMBL API Timeout

If ChEMBL downloads fail, try reducing `max_compounds` in your config.

### 2. ESM2 Model Download

First run downloads ~30MB transformer model. Requires internet connection.

### 3. Memory Issues

For large datasets (>5000 compounds), consider:
- Using XGBoost instead of Random Forest (`model_type: xgb_ensemble`)
- Reducing `n_ensemble` from 5 to 3
- Running on a machine with more RAM

## Getting Help

- Issues: https://github.com/frizelle-cheminfo/drug-discovery-prioritisation/issues
- Email: mitch@albion-os.com

## Citation

If you use this in your research:

```bibtex
@software{drug_discovery_prioritisation_2025,
  author = {Mitchell Frizelle},
  title = {Protein-Conditioned QSAR for Drug Discovery Prioritisation},
  year = {2025},
  url = {https://github.com/frizelle-cheminfo/drug-discovery-prioritisation}
}
```
