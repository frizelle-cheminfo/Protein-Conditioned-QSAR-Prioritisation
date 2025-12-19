# Quick Reference Card

One-page reference for common tasks.

## Installation

```bash
conda env create -f environment.yml
conda activate drug-prioritisation
```

## Run Tests

```bash
pytest tests/ -v
```

## Run Benchmark

```bash
python scripts/run_benchmark.py --config configs/case_study_erbb.yaml
```

## Score Compounds

```bash
python scripts/run_prioritisation.py \
  --compounds my_compounds.csv \
  --models-dir models/egfr/ \
  --primary-target EGFR \
  --off-targets ALK JAK2 \
  --output results.csv
```

## Config Template

```yaml
targets:
  - chembl_id: "CHEMBL203"
    name: "EGFR"

primary_target: "EGFR"
output_dir: "./outputs/my_experiment"
model_type: "rf_ensemble"
n_ensemble: 5
```

## Important Files

| File | Purpose |
|------|---------|
| `README.md` | Main documentation |
| `src/` | Reusable modules |
| `scripts/run_benchmark.py` | Full pipeline |
| `configs/` | Experiment configs |
| `tests/` | Unit tests |

## Common Commands

```bash
# Install in development mode
pip install -e .

# Run single test
pytest tests/test_features.py -v

# Format code
black src/ scripts/ tests/

# Check directory structure
ls -R
```

## Output Structure

```
outputs/case_study_erbb/
├── benchmarks/
│   ├── within_target_scaffold.csv
│   ├── leave_one_target_out.csv
│   └── enrichment.csv
├── plots/
│   ├── benchmark_comparison.png
│   └── enrichment_comparison.png
└── cleaned_data.csv
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ChEMBL timeout | Reduce `max_compounds` in config |
| Memory error | Use XGBoost, reduce ensemble size |
| Import error | Run `pip install -e .` from root |
| ESM2 download fails | Check internet, retry |

## Performance Tips

- Use `model_type: xgb_ensemble` for faster training
- Reduce `n_ensemble` from 5 to 3 for speed
- Use `max_compounds: 500` for quick tests

## Key Metrics

- **Spearman ρ** - Rank correlation (within-target: 0.64–0.74)
- **Enrichment@10%** - Active compound recovery (2–3× over random)
- **LOTO** - Cross-target generalisation (ρ ≈ 0.1, expected)

## Citation

```bibtex
@software{drug_discovery_prioritisation_2025,
  author = {Mitchell Frizelle},
  title = {Protein-Conditioned QSAR for Drug Discovery Prioritisation},
  year = {2025}
}
```
