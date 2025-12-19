# Protein-Conditioned QSAR for Drug Discovery Prioritisation

A practical system for rank-ordering small molecules in early-stage drug discovery campaigns, with explicit handling of multi-target selectivity, synthetic accessibility, and prediction uncertainty.

---

## Why This Exists

Hit identification in drug discovery generates thousands of compounds from high-throughput screens, fragment libraries, or virtual enumeration. The bottleneck isn't prediction accuracy—it's **decision-making under uncertainty**: which 20 compounds do you synthesise first?

This system addresses that by:
- Ranking compounds using multi-objective scoring (potency, selectivity, makeability, uncertainty)
- Providing explicit uncertainty quantification via ensemble disagreement
- Enabling protein-conditioned predictions for selectivity analysis
- Operating on public ChEMBL data, accepting its noise as representative of real-world heterogeneity

This is **not** a docking engine, binding mode predictor, or generative model. It is a lightweight triage tool for medicinal chemistry teams working under resource constraints.

---

## What It Does

### Core Capabilities

**Protein-Conditioned QSAR**
Combines ligand fingerprints (Morgan, physicochemical descriptors) with frozen protein embeddings (ESM2) to predict binding affinity (pIC50, pKi, pEC50). The protein embedding allows the same model to generalise across related targets.

**Multi-Target Selectivity Scoring**
Trains separate models for on-target and off-targets (e.g., EGFR vs. ERBB family), then computes selectivity as:
```
Selectivity = pActivity(on-target) - max(pActivity(off-targets))
```

**Uncertainty-Aware Ranking**
Uses 5-model ensembles (Random Forest or XGBoost) to estimate prediction uncertainty via standard deviation. Conservative scores penalise high-uncertainty predictions:
```
Conservative Score = μ - λσ  (λ = 0.5 default)
```

**Synthetic Accessibility Estimation**
Heuristic scoring based on molecular complexity (rings, stereocentres, molecular weight) to deprioritise difficult-to-make structures.

**Scaffold-Based Enumeration Support**
Designed to work with libraries generated from scaffold hops or R-group enumeration, where you need to rank 100–10,000 analogues.

---

## Use Cases

### 1. EGFR Inhibitor Campaign with ErbB Selectivity
**Goal:** Find EGFR-selective compounds that avoid ERBB2/ERBB4 (cardiotoxicity risk).

```python
TARGETS = [
    {"chembl_id": "CHEMBL203", "name": "EGFR"},
    {"chembl_id": "CHEMBL1824", "name": "ERBB2"},
    {"chembl_id": "CHEMBL2664", "name": "ERBB4"}
]
PRIMARY_TARGET = "EGFR"
OFF_TARGETS = ["ERBB2", "ERBB4"]
```

**Output:** Ranked list with selectivity scores, e.g., "Compound A: EGFR 8.2, ERBB2 6.1, selectivity +2.1".

### 2. Kinase Inhibitor with Safety Panel Avoidance
**Goal:** Optimise potency whilst minimising hERG, CYP3A4, and PgP liabilities.

```python
TARGETS = [
    {"chembl_id": "CHEMBL4822", "name": "JAK2"},
    {"chembl_id": "CHEMBL240", "name": "hERG"},
    {"chembl_id": "CHEMBL340", "name": "CYP3A4"}
]
PRIMARY_TARGET = "JAK2"
OFF_TARGETS = ["hERG", "CYP3A4"]
```

**Output:** Compounds ranked by combined score balancing JAK2 potency and ADMET avoidance.

### 3. Scaffold Analogue Prioritisation
**Goal:** 500 analogues enumerated from a pyrimidine scaffold—which 20 to synthesise?

```python
# Provide custom CSV with SMILES + predicted or known activity
DATA_MODE = "user_csv"
USER_CSV_PATH = "enumerated_analogues.csv"
```

**Output:** Top 20 ranked by potency prediction, uncertainty, and makeability.

---

## Case Studies Included

### Case Study 1: EGFR / ErbB Selectivity Panel
- **Targets:** EGFR, ERBB2, ERBB4, ALK, JAK2
- **Data:** ~1,000 compounds per target from ChEMBL (IC50)
- **Outputs:**
  - `benchmarks/within_target_scaffold.csv` — Per-target performance (scaffold splits)
  - `benchmarks/leave_one_target_out.csv` — Cross-target generalisation
  - `prioritised_compounds.csv` — Ranked EGFR inhibitors with selectivity scores
  - `plots/benchmark_comparison.png` — Spearman correlations across models
  - `report.pdf` — Summary PDF for medicinal chemistry review

**Key Finding:** Spearman ρ = 0.64–0.74 within-target (scaffold split), demonstrating practical rank-ordering ability. LOTO performance is poor (ρ ≈ 0.1), highlighting that protein embeddings alone don't enable true cross-family generalisation—this is expected and acceptable for the intended use case.

### Case Study 2: EGFR Potency + ADMET Panel
- **Targets:** EGFR, hERG, CYP3A4, CYP2D6
- **Objectives:** Maximise EGFR potency, minimise cardiotoxicity (hERG) and metabolic liabilities
- **Outputs:** Same structure as Case Study 1, with selectivity scores now representing safety margins

---

## Data Sources and Realism

**ChEMBL Public Data**
All training data comes from ChEMBL bioactivity records (IC50, Ki, EC50). This data is:
- **Heterogeneous:** Different assay formats, labs, conditions
- **Noisy:** ±0.5 log units is typical for replicate measurements
- **Incomplete:** Missing stereochemistry, tautomers, assay details

**Why This Is Acceptable**
Early discovery prioritisation is about **comparative ranking**, not absolute potency prediction. A model that consistently ranks known actives above inactives (enrichment@10% = 2.5×) is useful, even if RMSE = 0.8 pIC50 units.

Real-world company data would improve performance, but the system's architecture (ensemble uncertainty, selectivity scoring) remains valid.

---

## What This Is **Not**

### Not a Generative Model
This system ranks existing compounds or enumerated libraries. It does not propose novel structures. (Future: couple with REINVENT, Augmented Hill-Climb, or diffusion models.)

### Not a Docking Engine
No binding poses, no protein dynamics, no MM/GBSA rescoring. For **mechanism-of-action** hypotheses, use Glide, AutoDock, or GOLD. This tool assumes you already know your target and need to rank ligands.

### Not a Replacement for Experimental Validation
Predictions guide synthesis prioritisation—they don't replace biochemical assays. Think of this as a smart filter, not an oracle.

### Not a Solution for Kinetics or Dynamics
Predicts equilibrium binding (IC50/Ki), not kon/koff, residence time, or allosteric effects. For those, you need MD simulations or SPR/BLI experiments.

---

## Limitations

**1. Protein Embedding Quality**
Uses ESM2-8M (frozen), which captures sequence similarity but not conformational states. Better representations (AlphaFold2 embeddings, contact maps) would improve cross-target generalisation.

**2. Training Data Bias**
ChEMBL over-represents kinases, GPCRs, proteases. Performance on understudied targets (e.g., transcription factors) will be worse.

**3. No Explicit ADMET Models**
Makeability scoring is heuristic. For rigorous ADMET prediction, integrate SwissADME, pkCSM, or commercial tools.

**4. Scaffold Extrapolation**
LOTO results show poor generalisation to unseen protein families. This is a feature, not a bug—the system is designed for within-target or close-analogue ranking, not de novo hit discovery.

**5. No Active Learning**
Static models. For iterative optimisation, add Bayesian optimisation or Thompson sampling (future work).

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
│
├── notebooks/
│   └── Drug_Discovery_Prioritization_MVP_v3.ipynb  # Original exploratory notebook
│
├── src/
│   ├── data/
│   │   ├── chembl_downloader.py      # ChEMBL API client
│   │   ├── preprocessing.py          # SMILES validation, pActivity conversion
│   │   └── feature_engineering.py    # Fingerprints, descriptors, protein embeds
│   ├── models/
│   │   ├── ensemble.py               # RF/XGB ensemble training
│   │   ├── evaluation.py             # Metrics, enrichment, calibration
│   │   └── protein_embeddings.py     # ESM2 + fallback encoders
│   ├── scoring/
│   │   ├── selectivity.py            # Multi-target selectivity scoring
│   │   ├── makeability.py            # Synthetic accessibility heuristics
│   │   └── prioritisation.py         # Combined scoring function
│   └── utils/
│       ├── splits.py                 # Scaffold, LOTO splits
│       └── reporting.py              # PDF generation, plots
│
├── scripts/
│   ├── run_benchmark.py              # Reproduce benchmark results
│   ├── run_prioritisation.py         # Score new compound set
│   └── train_models.py               # Train models from config
│
├── configs/
│   ├── case_study_erbb.yaml          # EGFR selectivity config
│   └── case_study_admet.yaml         # ADMET panel config
│
├── outputs/                          # Generated results (not in git)
│   ├── case_study_erbb/
│   │   ├── benchmarks/
│   │   ├── plots/
│   │   ├── prioritised_compounds.csv
│   │   └── report.pdf
│   └── case_study_admet/
│       └── ...
│
└── tests/
    ├── test_features.py
    ├── test_models.py
    └── test_scoring.py
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/frizelle-cheminfo/drug-discovery-prioritisation.git
cd drug-discovery-prioritisation

# Create conda environment
conda env create -f environment.yml
conda activate drug-prioritisation

# Or use pip
pip install -r requirements.txt
```

**Key Dependencies:**
- RDKit (cheminformatics)
- scikit-learn, XGBoost (ML)
- transformers, torch (ESM2 protein embeddings)
- chembl_webresource_client (data download)
- matplotlib, seaborn (visualisation)

---

## Quickstart

### Run a Pre-Configured Case Study

```bash
python scripts/run_benchmark.py --config configs/case_study_erbb.yaml
```

This will:
1. Download ChEMBL data for EGFR, ERBB2, ALK, JAK2
2. Train ensemble models (ligand-only + protein-conditioned)
3. Run scaffold split and LOTO benchmarks
4. Generate enrichment plots and PDF report
5. Save outputs to `outputs/case_study_erbb/`

### Score Your Own Compound Library

```python
from src.scoring.prioritisation import PrioritisationPipeline

# Load pre-trained models
pipeline = PrioritisationPipeline.from_checkpoint("models/egfr_ensemble.pkl")

# Score new compounds
compounds = ["CCN(CC)C(=O)Nc1ccc2ncnc(Nc3cccc(Br)c3)c2c1", ...]  # Your SMILES
results = pipeline.score(
    compounds,
    primary_target="EGFR",
    off_targets=["ERBB2", "ERBB4"],
    weights={"potency": 0.5, "selectivity": 0.3, "makeability": 0.2}
)

results.to_csv("my_prioritised_compounds.csv")
```

---

## Future Directions

**Better Protein Representations**
Replace ESM2-8M with AlphaFold2 structure embeddings or graph neural networks on protein pockets. This would improve cross-target generalisation and enable allosteric site modelling.

**Integration with Generative Models**
Use prioritisation scores as reward functions for REINVENT, MolGPT, or diffusion models. Close the loop: generate → prioritise → synthesise → measure → retrain.

**Active Learning**
Add Bayesian optimisation or Thompson sampling to iteratively select compounds for synthesis, maximising information gain per experiment.

**Dynamics-Aware Scoring**
Incorporate MD simulation metrics (RMSF, pocket flexibility) or residence time predictors for targets where kinetics matter (e.g., kinases, proteases).

**World Models for Drug Discovery**
Train a latent dynamics model of SAR landscapes, enabling counterfactual reasoning: "If I make this substitution, what happens to selectivity?"

---

## Contributing

This is a research-stage tool. Contributions welcome for:
- Additional protein embedding methods (ProtBERT, Ankh, etc.)
- ADMET model integration (hERG, Caco-2, CYP)
- Deployment as REST API or web interface
- Benchmarking on proprietary datasets

---

## Citation

If you use this in your research or startup, please cite:

```bibtex
@software{drug_discovery_prioritisation_2025,
  author = {Mitchell Frizelle},
  title = {Protein-Conditioned QSAR for Drug Discovery Prioritisation},
  year = {2025},
  url = {https://github.com/yourusername/drug-discovery-prioritisation}
}
```

---

## Licence

MIT Licence. See `LICENSE` for details.

**Disclaimer:** This software is for research purposes only. Not validated for clinical use. Predictions should inform, not replace, experimental validation.

---

## Contact

For questions, collaboration enquiries 
- Email: mitch@albion-os.com


---

## Acknowledgements

- **ChEMBL** for open bioactivity data
- **ESM2** (Meta) for protein language models
- **RDKit** community for cheminformatics tools

