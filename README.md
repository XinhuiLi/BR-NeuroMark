# Schizophrenia Functional Network Connectivity - Multiverse Analysis

Multiverse / specification-curve analysis (Steegen et al., 2016; Simonsohn et al., 2020) varies analytic choices across **five forks** (connectivity, confound, reduction, classifier, domain) and tests robustness of three hypotheses. Operational definitions and **what the full run supports** are summarized below; counts come from `results/multiverse_full/multiverse_results.csv` for the **480**-specification grid.

### Hypotheses (what each test measures)

- **H1 — Latent vs edge classification.** For each specification, the same FC **edge features** support two nested-CV pipelines: **all edges** vs **edges projected to *k* latent components** (FA, ICA, PCA, or NMF; *k* chosen in-fold). The multiverse outcome is **ΔAUC = mean outer-fold ROC-AUC(latent) − mean outer-fold ROC-AUC(edges)**. A specification counts as **favourable** if **ΔAUC > 0** (latent strictly better on average). Rows with `reduction = none` have no latent arm and are excluded from H1 summaries (**384** evaluable specs in the saved full run).

- **H2 — Between-domain vs within-domain effect sizes.** Per edge, **Cohen’s *d*** for SCZ vs HC is computed. The statistic is **Δ mean|*d*| = mean(|*d*| on between-domain edges) − mean(|*d*| on within-domain edges)** (domain masks follow the chosen **D5** granularity). A **label permutation** of ICN domain assignments yields a directional permutation *p*. A spec is **favourable** if ***p* < 0.05 and Δ mean|*d*| > 0**. H2 uses the (possibly confound-adjusted) edge matrix only, so summaries collapse duplicate classifier/reduction rows to **24** unique D1×D2×D5 pipelines.

- **H3 — Between- vs within-domain loading mass.** After fitting the chosen reducer on scaled edges, each component’s **mean |loading|** is compared **between-domain vs within-domain**. Across components, paired differences are tested with a **Wilcoxon signed-rank** on those differences (see `fbirn_experiment/multiverse.py`). A spec is **favourable** only if **Wilcoxon *p* < 0.05 and a strict majority of components have between-domain loading mass greater than within-domain loading mass**. Like H1, H3 requires a latent decomposition (**384** evaluable specs with `reduction ≠ none`).

### Key takeaways from `results/multiverse_full`

| Hypothesis | Supported in the multiverse sense? | Typical “winning” settings (from `mv_conditional_robustness.csv`) |
|------------|-------------------------------------|-------------------------------------------------------------------|
| **H1** | **No** as a default claim: only **18.8%** of latent specs have ΔAUC > 0; **median ΔAUC = −0.0319** (edges often win on average). The joint binomial test still flags **more** positive-Δ specs than a strict global null, so some analytic paths favour latent, but they are a **minority**. | **ICA** reduction (**54.2%** favourable within ICA rows) vs **FA/PCA** (**8.3%**/**2.1%**). **ComBat** confounds (**26.6%**) vs **none** (**10.9%**). **Partial correlation** connectivity (**29.2%**) vs **mutual information** (**8.3%**). Domain granularity makes little difference (**18.8%** for both D5 levels). |
| **H2** | **Mixed and estimator-dependent**: **50.0%** of the 24 unique H2 pipelines are favourable; **median Δ mean\|d\| = 0.0079** (> 0). | **Pearson (Fisher z)** and **Spearman** are **100%** favourable; **partial correlation** and **mutual information** are **0%** favourable. Confound strategy and domain granularity are each **50%** favourable. |
| **H3** | **Limited direction-aware support**: **26.0%** of latent specs are significant and have a majority of components with between > within loading mass; **median *p* = 0.0543**. | **Factor analysis** (**45.8%**) and **PCA** (**33.3%**) are more often favourable than **ICA/NMF** (**12.5%** each) under this direction-aware definition. **Spearman** (**41.7%**) and **Pearson** (**37.5%**) exceed **partial correlation** (**25.0%**) and **mutual information** (**0%**). |

**Joint specification-count test** (`figures/mv_joint_permutation_test.csv`): for α = 0.05, the number of favourable specs **exceeds** the binomial null for **all three** hypotheses (including H1), which is consistent with **some** sensitivity of the global null to multiple testing structure—not with blanket latent superiority for H1.

---

## Full multiverse in this repo (`results/multiverse_full/`)

A **full factorial** run is saved under `results/multiverse_full/`: **480** specifications in `multiverse_results.csv`, from the grid **4×3×5×4×2** (confound strategies **`none`**, **`ols`**, and **`combat`**).

Tables and figures use rows with finite outcomes. H2 is collapsed to **24** unique D1×D2×D5 pipelines because classifier and reduction do not affect the edge-level H2 statistic; H1/H3 have **384** latent-reduction rows, i.e. `reduction ≠ none`.

### Fork grid (full default)

| Fork | Levels |
|------|--------|
| **D1** Connectivity | `pearson_z`, `spearman`, `partial_corr`, `mutual_info` |
| **D2** Confound | `none`, `ols`, `combat` |
| **D3** Reduction | `none`, `fa`, `ica`, `pca`, `nmf` |
| **D4** Classifier | `elasticnet`, `logistic_l2`, `svm_linear`, `rf` |
| **D5** Domain | `domain_7`, `subdomain_14` |
| **Count** | **480** = 4×3×5×4×2 |

Full run (480-spec grid): `python -m fbirn_experiment.cli multiverse --out results/multiverse_full --confound-strategies none ols combat --n-jobs -1` (completed specs are skipped via `specs/*.json` checkpoints).

### Robustness summary (full grid)

Source: `results/multiverse_full/figures/mv_robustness_summary.csv` (same logic as `multiverse_figures.robustness_table()` on the filtered dataframe).

| Hypothesis | Specs (evaluable) | Favourable | % | Median effect |
|------------|-------------------|------------|---|---------------|
| H1: Latent > edges | 384 | 72 | 18.8% | −0.0319 |
| H2: Between > within | 24 | 12 | 50.0% | 0.0079 |
| H3: Between loading advantage | 384 | 100 | 26.0% | 0.0543 median *p* |

Joint binomial test (`results/multiverse_full/figures/mv_joint_permutation_test.csv`): for each hypothesis the count of favourable specs exceeds the α = 0.05 binomial null (*n* as in the table). Interpretation of H1 vs that test is spelled out in **Key takeaways** above.

---

## Study design

![Study design](figures/study_design.png)

---

## Data

Covariate / confound distributions (`figures/confound_distributions.png`) and group-mean functional connectivity (Pearson *z*) for controls vs schizophrenia patients (`figures/mean_fnc_pearson_z_hc_sz.png`). Regenerate with `python -m fbirn_experiment.cli plot-confounds` and `python -m fbirn_experiment.cli mean-fnc-matrices`.

![Confound distributions](figures/confound_distributions.png)

![Mean FNC (Pearson z), HC and SCZ](figures/mean_fnc_pearson_z_hc_sz.png)

---

## Figures (full)

Specification curves and raincloud “forest” plots live in `results/multiverse_full/figures/`:

![H1 spec curve](results/multiverse_full/figures/mv_spec_curve_h1.png)

![H2 spec curve](results/multiverse_full/figures/mv_spec_curve_h2.png)

![H3 spec curve](results/multiverse_full/figures/mv_spec_curve_h3.png)

![H1 forest](results/multiverse_full/figures/mv_forest_h1.png)

![H2 forest](results/multiverse_full/figures/mv_forest_h2.png)

![H3 forest](results/multiverse_full/figures/mv_forest_h3.png)

---

## Quick run (`--mini`, 48 specs)

For a small grid, use:

```bash
python -m fbirn_experiment.cli multiverse --mini --out results/multiverse
```

| Fork | `--mini` levels |
|------|-----------------|
| **D1** | `pearson_z`, `spearman` |
| **D2** | `none`, `ols` |
| **D3** | `none`, `fa`, `ica` |
| **D4** | `logistic_l2`, `svm_linear` |
| **D5** | `domain_7`, `subdomain_14` |

Mini figures/CSVs: `results/multiverse/figures/` (not the same numbers as the full run above).

---

## CLI

From the repository root, put the package on `PYTHONPATH` (e.g. `export PYTHONPATH="$PWD"`) or install the project in editable mode.

### Single-pipeline experiment (`run`)

Default pipeline: one fixed analytic specification — nested CV for **H1** (all FNC edges with **L2 logistic regression** vs **ICA** latent features; add **`--h1-include-fa`** for FA in H1), **H2** domain permutation test, **H3** **ICA** loadings on edges (FastICA; automatic *k* via reconstruction MSE unless **`--h3-no-bic`**; use **`--h3-fa`** for factor analysis + BIC *k*) — implemented in `fbirn_experiment/pipeline.py`.

```bash
python -m fbirn_experiment.cli run --out results/fbirn_icn_run
```

Defaults use `fbirn_experiment.config` paths for time courses (`--tc`) and labels (`--labels`) when those files exist. Common options: `--confounds-csv`, `--no-confounds`, `--outer-splits`, `--inner-splits`, `--no-figures`, `--no-save`, `--h1-include-fa`.

**H1 stability and interpretability (single run):** `artifacts/h1_stability_tests.json` — Levene and Fligner tests on outer-fold AUCs. `artifacts/h1_interpretability_meta.json` and `h1_interpretability_coefs.npz` — full-sample refits at **median** nested-CV hyperparameters. Figures: `figures/h1_auc_stability_violin.png`, `h1_edge_top_coefficients.png`, `h1_ica_latent_interpretability.png`, and `h1_fa_latent_interpretability.png` only if H1 included FA (`--h1-include-fa`). Coefficients are exploratory, not nested-CV unbiased.

**Regenerate H1 latent figures from artifacts only:**

```bash
python -m fbirn_experiment.cli regen-h1-latent-figs --run-dir results/fbirn_icn_run
```

### Multiverse analysis (`multiverse`)

```bash
# Mini (48 specs)
python -m fbirn_experiment.cli multiverse --mini --out results/multiverse

# Full factorial (480 specs: none + ols + combat); parallel workers; resume via specs/*.json
python -m fbirn_experiment.cli multiverse --out results/multiverse_full --confound-strategies none ols combat --n-jobs -1

# Custom slice
python -m fbirn_experiment.cli multiverse \
  --connectivity pearson_z spearman \
  --confound-strategies none ols \
  --reductions none fa ica \
  --classifiers logistic_l2 \
  --granularities domain_7 subdomain_14
```

**Regenerate multiverse figures from an existing `multiverse_results.csv` (no recompute):**

```bash
python -m fbirn_experiment.cli regen-multiverse-figs --multiverse-dir results/multiverse_full
# or: --results-csv path/to/multiverse_results.csv [--figures-dir path/to/out]
```

Multiverse flags: `--out`, `--no-figures`, `--synthetic`, `--h2-perm`, `--n-jobs`, plus per-fork overrides listed earlier.

---

## Implementation notes

- **Checkpoint resume:** completed `specs/{spec_id:04d}.json` are skipped.
- **Run provenance:** `run_manifest.json` records array shapes, class/domain counts, confound file hash, fork levels, seeds/CV settings, package versions, and git state for new multiverse runs.
- **Parallelism:** `joblib` + `n_jobs`.
- **Connectivity:** Pearson / Spearman / partial correlation (Ledoit–Wolf) / mutual information — see `connectivity.py`.
- **ComBat:** optional confound level in `multiverse.py` (`--confound-strategies combat`) if `neuroCombat` is installed; included in the saved **480**-spec grid above.
- **Multiverse figure labels:** human-readable fork levels (e.g. “Mutual information”) via `multiverse_figures.format_fork_level`.

---

## References

1. Steegen et al. (2016). Multiverse analysis. *Perspectives on Psychological Science*.
2. Simonsohn et al. (2020). Specification curve analysis. *Nature Human Behaviour*.
3. Del Giudice & Gangestad (2021). Traveler’s Guide to the Multiverse. *AMPPS*.
4. Burkhardt & Gießing (2024). COMET toolbox. *bioRxiv*.
5. Kristanto et al. (2024). FC preprocessing multiverse review. *Neurosci Biobehav Rev*.

Full fork splits (conditional robustness): `results/multiverse_full/figures/mv_conditional_robustness.csv`.
