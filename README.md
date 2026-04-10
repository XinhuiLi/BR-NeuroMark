# Schizophrenia FNC — Multiverse Analysis

Multiverse / specification-curve analysis (Steegen et al., 2016; Simonsohn et al., 2020) varies analytic choices across **five forks** (connectivity, confound, reduction, classifier, domain) and tests robustness of three hypotheses (H1: classification edges vs latent; H2: between- vs within-domain effect sizes; H3: factor-loading structure).

---

> **Results in this repo** (`results/multiverse/`, figures under `results/multiverse/figures/`) are produced with the **`--mini`** setting — **48 specifications**, not the full factorial. See the table below before generalizing numbers to “the multiverse” as a whole.

## Mini vs full multiverse

| Fork | `--mini` | Full default\* |
|------|----------|----------------|
| **D1** Connectivity | `pearson_z`, `spearman` | also `partial_corr`, `mutual_info` |
| **D2** Confound | `none`, `ols` | also `combat` |
| **D3** Reduction | `none`, `fa`, `ica` | also `pca`, `nmf` |
| **D4** Classifier | `logistic_l2`, `svm_linear` | also `elasticnet`, `rf` |
| **D5** Domain | `domain_7`, `subdomain_14` | same |
| **Count** | **48** = 2×2×3×2×2 | **480** = 4×3×5×4×2 |

\*Full run: `python -m fbirn_experiment.cli multiverse` from the repo root (with `fbirn_experiment` on `PYTHONPATH`) and no `--mini` or narrowing flags; `enumerate_multiverse()` fills missing lists from defaults in `multiverse.py`.

**Why mini omits some levels:** Smaller grid for routine runs; `--mini` drops ComBat so results are not duplicated when `neuroCombat` is absent. Use the full grid + optional CLI filters for broader coverage.

## Robustness summary (mini)

Source: `results/multiverse/figures/mv_robustness_summary.csv`.

| Hypothesis | Specs (evaluable) | Favourable | % | Median effect |
|------------|-------------------|------------|---|---------------|
| H1 Latent > edges | 32 | 10 | 31.2% | −0.010 |
| H2 Between > within | 48 | 48 | **100%** | 0.037 |
| H3 Loading advantage | 32 | 26 | **81.2%** | 0.010 |

Joint binomial test (`mv_joint_permutation_test.csv`): counts exceed the α = 0.05 null for all three hypotheses; interpret H1 together with FA vs ICA (H1 is decomposition-driven, not classifier-driven in this grid).

---

## Figures (mini)

Specification curves and raincloud “forest” plots:

![H1 spec curve](results/multiverse/figures/mv_spec_curve_h1.png)

![H2 spec curve](results/multiverse/figures/mv_spec_curve_h2.png)

![H3 spec curve](results/multiverse/figures/mv_spec_curve_h3.png)

![H1 forest](results/multiverse/figures/mv_forest_h1.png)

![H2 forest](results/multiverse/figures/mv_forest_h2.png)

![H3 forest](results/multiverse/figures/mv_forest_h3.png)

---

## CLI

From the repository root, put the package on `PYTHONPATH` (e.g. `export PYTHONPATH="$PWD"`) or install the project in editable mode.

### Single-pipeline experiment (`run`)

Default pipeline: one fixed analytic specification — nested CV for H1 (edges vs FA vs ICA), H2 domain permutation test, H3 factor loadings — implemented in `fbirn_experiment/pipeline.py`.

```bash
python -m fbirn_experiment.cli run --out results/fbirn_icn_run
```

Defaults use `fbirn_experiment.config` paths for time courses (`--tc`) and labels (`--labels`) when those files exist. Common options: `--confounds-csv`, `--no-confounds`, `--outer-splits`, `--inner-splits`, `--no-figures`, `--no-save`.

### Multiverse analysis (`multiverse`)

```bash
# Mini (48 specs); figures + CSVs under results/multiverse/
python -m fbirn_experiment.cli multiverse --mini

# Full factorial (480 specs); --n-jobs -1; completed specs are skipped via JSON checkpoints
python -m fbirn_experiment.cli multiverse --n-jobs -1

# Custom slice
python -m fbirn_experiment.cli multiverse \
  --connectivity pearson_z spearman \
  --confound-strategies ols \
  --reductions none fa ica \
  --classifiers logistic_l2 \
  --granularities domain_7 subdomain_14
```

Multiverse flags: `--out`, `--no-figures`, `--synthetic`, `--h2-perm`, `--n-jobs`, plus per-fork overrides listed above.

---

## Implementation notes

- **Checkpoint resume:** completed `specs/{spec_id:04d}.json` are skipped.
- **Parallelism:** `joblib` + `n_jobs`.
- **Connectivity:** Pearson / Spearman / partial correlation (Ledoit–Wolf) / mutual information — see `connectivity.py`.
- **ComBat:** CV-aware path in `multiverse.py`; install `neuroCombat` for true ComBat on full grids.

---

## References

1. Steegen et al. (2016). Multiverse analysis. *Perspectives on Psychological Science*.
2. Simonsohn et al. (2020). Specification curve analysis. *Nature Human Behaviour*.
3. Del Giudice & Gangestad (2021). Traveler’s Guide to the Multiverse. *AMPPS*.
4. Burkhardt & Gießing (2024). COMET toolbox. *bioRxiv*.
5. Kristanto et al. (2024). FC preprocessing multiverse review. *Neurosci Biobehav Rev*.

Full tables (conditional robustness, all fork splits): `results/multiverse/figures/mv_conditional_robustness.csv`.
