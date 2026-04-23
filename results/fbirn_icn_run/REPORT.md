# Latent Connectivity Factors in Schizophrenia: An Analysis of NeuroMark ICN Functional Network Connectivity

## 1. Introduction

Schizophrenia is characterized by disrupted functional brain connectivity, yet the high dimensionality of functional network connectivity (FNC) matrices makes it difficult to identify the most informative patterns of dysconnectivity. This study compares **patient vs. control classification** using **all FNC edges** (dense **L2 logistic regression**) to classification on **FastICA** latent features, and separately characterizes **edge-level group effects** (H2) and **exploratory ICA loadings** by domain structure (H3).

We use intrinsic connectivity network (ICN) time courses estimated via the NeuroMark 2.2 framework (Iraji et al., 2023) from the Function Biomedical Informatics Research Network (FBIRN) resting-state fMRI dataset. ICN domain–subdomain assignments follow the NeuroMark 2.2 taxonomy, yielding 14 distinct domain–subdomain labels across 7 functional domains. Three hypotheses are tested:

- **H1 (ICA vs. edges):** Patient vs. control classification using **all FNC edges** with **L2 logistic regression** is compared to classification on **FastICA** latent features (with nested cross-validation and fold-wise confound removal for H1).
- **H2 (Between- vs. within-domain effects):** Between-domain (long-range) FNC edges exhibit larger group effect sizes than within-domain (short-range) edges, consistent with disrupted large-scale integration in schizophrenia.
- **H3 (Latent loading structure):** Exploratory **FastICA** on confound-regressed edges yields **independent components** whose absolute loadings are summarized by edge class (between- vs. within-domain) and by literature-based hypothesis domain pairs (same pairs as in the code; exploratory, full-sample fit).

## 2. Dataset

| Property | Value |
|---|---|
| Dataset | FBIRN (Function Biomedical Informatics Research Network) |
| Subjects | 363 (182 schizophrenia patients, 181 healthy controls) |
| Modality | Resting-state fMRI |
| ICN template | NeuroMark 2.2 (Iraji et al., 2023, *Human Brain Mapping*) |
| Number of ICNs | 105 |
| Domain–subdomain labels | 14 unique labels across 7 domains (see Table 1) |
| Time points per subject | 157 |
| FNC edges per subject | 5,460 (upper triangle of 105 × 105) |
| Labels | Binary (1 = SZ, 0 = HC) |
| Data files | `data/FBIRN_ICN_TC_v2.2.npy` (363 × 157 × 105), `data/FBIRN_label.npy` (363,) |
| Confounds regressed | Age, sex, race, site, head motion (13 regressors incl. intercept) |

**Table 1. NeuroMark 2.2 domain–subdomain labels.**

| Label | Domain | Subdomain | # ICNs |
|---|---|---|---|
| CB | Cerebellar | Cerebellar | 13 |
| VI-OT | Visual | Occipitotemporal | 6 |
| VI-OC | Visual | Occipital | 6 |
| PL | Paralimbic | Paralimbic | 11 |
| SC-EH | Subcortical | Extended Hippocampal | 3 |
| SC-ET | Subcortical | Extended Thalamic | 6 |
| SC-BG | Subcortical | Basal Ganglia | 9 |
| SM | Sensorimotor | Sensorimotor | 14 |
| HC-IT | Higher Cognition | Insular-Temporal | 7 |
| HC-TP | Higher Cognition | Temporoparietal | 5 |
| HC-FR | Higher Cognition | Frontal | 10 |
| TN-CE | Triple Network | Central Executive | 3 |
| TN-DM | Triple Network | Default Mode | 8 |
| TN-SA | Triple Network | Salience | 4 |

## 3. Methods

### 3.1 FNC Construction and Confound Regression

For each subject, a 105 × 105 Pearson correlation matrix was computed from the ICN time courses (157 time points × 105 ICNs). Correlations were Fisher z-transformed for variance stabilization. The strict upper triangle was vectorized to yield 5,460 FNC edges per subject.

**Confound regression.** A design matrix was constructed from five confounding variables: age (continuous), sex (one-hot), race (one-hot: Black, Other, White), site (one-hot: 7 FBIRN sites), and head motion (continuous mean framewise displacement), plus an intercept — yielding 13 total regressors. For H2 and H3 analyses, confound regression was performed on the full sample via OLS (residualization). For H1 classification, confounds were regressed within each cross-validation fold: OLS coefficients were fit on training data only and applied to both training and test sets to prevent data leakage.

### 3.2 H1: Classification — Edges vs. ICA

A **5-fold stratified nested cross-validation** design compared **two** classification pipelines (factor analysis is **not** included in the default H1 grid; it can be enabled in the codebase with `--h1-include-fa`):

1. **Edges + L2 logistic regression:** All 5,460 z-scored FNC edges were standardized and classified with **L2-penalized logistic regression** (`sklearn.linear_model.LogisticRegression`, `lbfgs`). Inner CV tuned the inverse regularization strength *C* over a log-spaced grid.
2. **ICA + L2 logistic regression:** Within each outer training fold, the number of FastICA components *k* was selected by minimizing **reconstruction MSE** over *k* ∈ {5, 10, …, 50}. ICA-transformed features were classified with the same L2 logistic family; inner CV tuned *C*.

**AUC comparison statistics** (ICA vs. edges):
- Out-of-fold bootstrap (10,000 resamples) with 95% CI for ΔAUC (ICA − edges)
- Label-permutation test (5,000 permutations) for ΔAUC under the null
- Paired Wilcoxon signed-rank test on per-fold AUCs

### 3.3 H2: Between- vs. Within-Domain Effect Sizes

Each of the 5,460 edges was assigned a between-domain or within-domain label based on the NeuroMark 2.2 domain–subdomain membership (14 labels). Cohen's *d* (SZ − HC) was computed per edge on confound-regressed FNC. The test statistic was:

\[
\Delta = \text{mean}(|d|_{\text{between}}) - \text{mean}(|d|_{\text{within}})
\]

A **domain-label permutation test** (2,000 permutations) shuffled domain assignments across ICNs and re-computed Δ under the null to derive a two-sided *p*-value.

### 3.4 H3: Exploratory ICA Loadings by Edge Class

**FastICA** (`sklearn.decomposition.FastICA`, unit-variance whitening) was fit on all subjects' **z-scored, confound-regressed** FNC edges (full sample; exploratory). The number of components *k* was selected by minimizing **reconstruction MSE** over *k* ∈ {5, 10, …, 50} (same grid as H1 ICA branch). For each component, the mean absolute entry of the **unmixing matrix** (`components_`, edge × component projection) was computed separately for between-domain and within-domain edges. A paired Wilcoxon signed-rank test across components assessed whether between-domain magnitudes systematically differed from within-domain magnitudes.

**Note:** ICA loadings are on a different scale than Gaussian FA loadings; magnitudes should be interpreted **relative to each other** and to domain-pair contrasts, not compared numerically to older FA-based reports.

#### 3.4.1 Hypothesis Domain-Pair Analysis

Nine schizophrenia-relevant domain interactions were selected from the literature to test whether specific inter-domain edges drive the most discriminative latent factors. For each pair, the mean absolute loading across the relevant edges was computed per factor and averaged across all factors. These pair-specific loadings were compared against overall between-domain and within-domain baselines.

**Table 2. Hypothesis domain pairs and supporting literature.**

| Pair | Rationale | Key references |
|---|---|---|
| TN-DM × TN-CE | DMN–Central Executive anticorrelation | Li et al. 2019 *Front Psychiatry* 10:482; Littow et al. 2015 *Front Psychiatry* 6:26 |
| TN-DM × HC-IT | DMN–Insular/Temporal (auditory hallucinations) | Marino & Spironelli 2022 *J Psychiatr Res* (PMID:35981441); Weber et al. 2020 *Front Psychiatry* 11:227 |
| SC-ET × TN-CE | Thalamic–Central Executive disconnection | Woodward et al. 2012 *Am J Psychiatry* 169:1092 (PMID:23032387); Giraldo-Chica & Woodward 2017 *Schizophr Res* 180:58 |
| SC-ET × TN-DM | Thalamo-cortical loop disruption | Woodward et al. 2012 *Am J Psychiatry* 169:1092 (PMID:23032387); Ferri et al. 2018 *Sci Rep* 8:3451 |
| TN-CE × SM | Central Executive–Sensorimotor disintegration | Kaufmann et al. 2015 *Schizophr Bull* 41:1326 (PMID:25943122) |
| HC-IT × SM | Insular/Temporal–Sensorimotor processing disruption | Avram et al. 2018 *Neuropsychopharmacology* 43:2462; Kaufmann et al. 2015 *Schizophr Bull* 41:1326 |
| TN-DM × VI-OC | DMN–Visual (occipital) hypoconnectivity | Dong et al. 2018 *Sci Rep* 8:14655; Sendi et al. 2021 *Schizophr Res* 228:103 |
| TN-SA × TN-DM | Salience–DMN switching (triple network model) | Palaniyappan & Liddle 2012 *JNNP* 83:558; Menon 2011 *Trends Cogn Sci* 15:483 |
| TN-SA × TN-CE | Salience–Central Executive engagement | Menon 2011 *Trends Cogn Sci* 15:483; Manoliu et al. 2014 *Schizophr Bull* 40:428 |

### 3.5 Software and Reproducibility

All analyses used the `fbirn_experiment` Python package with scikit-learn for classification and decomposition, SciPy for statistical tests, and NumPy for numerical operations. The full pipeline is executable via:

```bash
python -m fbirn_experiment.cli run --out results/fbirn_icn_run
```

## 4. Results

### 4.1 H1: Classification Performance

| Model | AUC (mean ± SD) | Latent dimensions (per outer fold) |
|---|---|---|
| Edges (L2 logistic) | 0.823 ± 0.060 | — (5,460 features) |
| ICA + L2 logistic | **0.838 ± 0.059** | 50, 50, 50, 50, 50 (reconstruction MSE) |

Both models achieved high discriminative performance (AUC > 0.82) after confound regression. **ICA was numerically higher on mean outer-fold AUC** than the edge-based logistic model. The formal **ICA vs. edges** comparison (ΔAUC = ICA − edges, out-of-fold) did **not** reach *p* < 0.05 on any of the three auxiliary tests:

| Comparison | ΔAUC (ICA − edges) | Bootstrap *p* (two-sided) | 95% CI | Perm. *p* | Wilcoxon *p* (fold AUCs) |
|---|---|---|---|---|---|
| ICA vs. Edges | +0.0044 | 0.988 | [−0.0258, 0.0358] | 0.792 | 1.000 |

**Interpretation:** With the current default (edges as **dense L2 logistic** rather than elastic net), ICA features matched or slightly exceeded mean AUC while using 50 components vs. 5,460 edges, but uncertainty intervals are wide and tests do not support a definitive superiority of either pipeline.

**Stability (fold-to-fold AUC spread):** Levene *p* = 0.854 and Fligner–Killeen *p* = 0.543 on the two vectors of outer-fold AUCs (see `artifacts/h1_stability_tests.json`), giving **no evidence** that the two models differ in fold-to-fold dispersion of AUC.

#### Per-Fold Tuned Hyperparameters

**Edges (L2 logistic):** inner-CV *C* ∈ {0.1, 17.78, 100} pattern across folds (see `artifacts/h1_best_params_edges.json`).

**ICA + L2 logistic:** Reconstruction MSE selected *k* = **50** (search upper bound) in **all** five outer folds; inner-CV *C* alternated between **0.1** and **0.562** (`artifacts/h1_best_params_ica.json`).

### 4.2 H2: Between- vs. Within-Domain Effect Sizes

| Metric | Value |
|---|---|
| Mean \|Cohen's *d*\|, between-domain | 0.223 |
| Mean \|Cohen's *d*\|, within-domain | 0.202 |
| Observed Δ (between − within) | 0.0205 |
| Permutation *p* (two-sided, 2,000 perms) | **0.040** |

**Interpretation:** After confound regression, between-domain edges showed a significantly larger mean absolute effect size than within-domain edges (Δ ≈ 0.021, *p* = 0.040). This supports H2: group differences are more pronounced for **long-range** (between-domain) connections. NeuroMark 2.2 domain–subdomain labels are used throughout.

### 4.3 H3: ICA Loading Structure (Exploratory)

Reconstruction-MSE selection on the full sample chose **50 FastICA components** (*k* ∈ {5, 10, …, 50}; best MSE ≈ 0.330, `artifacts/h3_component_selection.json`). For each component, the mean absolute loading was averaged over between-domain vs. within-domain edges (`artifacts/h3_factor_loading_summary.csv`).

**Global edge-class contrast:** Across the 50 components, mean |loading| was **0.00165** (between) vs. **0.00171** (within) — i.e. slightly **lower** between-domain than within-domain on average. Between-domain exceeded within-domain in **20 of 50** components. A paired Wilcoxon signed-rank test on the 50 paired means gave ***p* = 0.088** (two-sided), so the **FA-era “between > within everywhere” signature is not replicated** under this ICA parameterization at α = 0.05.

**Example components (illustrative):**

| Component | Mean \|loading\|, between | Mean \|loading\|, within |
|---|---:|---:|
| 0 | 0.00164 | 0.00201 |
| 1 | 0.00163 | 0.00203 |
| 27 | **0.00107** | **0.00088** |
| 32 | **0.00182** | 0.00143 |

(All 50 rows are in the CSV; magnitudes are ~10⁻³ because ICA `components_` are not calibrated like FA loadings.)

**Interpretation:** H3 in this run is **exploratory** and **scale-dependent**. With **k = 50** at the MSE grid ceiling, ICA may still be over-parameterized for a stable edge-class contrast; the hypothesis-domain-pair view below is more informative for relative ranking of pairs than for absolute effect size.

#### 4.3.1 Hypothesis Domain-Pair Loadings (ICA)

Mean |loading| per hypothesis pair, **averaged across all 50 ICA components** (`artifacts/h3_domain_pair_loading_summary.csv`). Overall component-mean baselines: between-domain **0.00165**, within-domain **0.00171** (same as above).

| Domain pair | # Edges | Mean \|loading\| (avg over components) |
|---|---|---:|
| TN-DM × TN-CE | 24 | 0.00182 |
| TN-SA × TN-CE | 12 | 0.00174 |
| TN-DM × VI-OC | 48 | 0.00174 |
| TN-SA × TN-DM | 32 | 0.00173 |
| TN-CE × SM | 42 | 0.00172 |
| HC-IT × SM | 98 | 0.00169 |
| SC-ET × TN-DM | 48 | 0.00168 |
| TN-DM × HC-IT | 56 | 0.00166 |
| SC-ET × TN-CE | 18 | 0.00159 |

**Interpretation:** Under ICA, **triple-network and DMN–visual** pairs sit at the **top** of the hypothesis-pair ranking (absolute scale ~1.8×10⁻³), while **thalamic–central executive (SC-ET × TN-CE)** is **lowest** in this average ranking — a different emphasis than in the older FA-based report and consistent with ICA not preserving the same latent geometry as FA. Literature citations for each pair are unchanged (Table 2 / `h3_test.py`).

### 4.4 Summary of Hypothesis Tests

| Hypothesis | Supported? | Key statistic |
|---|---|---|
| **H1:** ICA outperforms edges | **No** (not significant; ICA slightly higher mean AUC) | ΔAUC (ICA − edges) ≈ +0.004; bootstrap / perm / Wilcoxon *p* > 0.05 |
| **H2:** Between-domain > within-domain \|*d*\| | **Yes** | Δ ≈ 0.021; perm. *p* = 0.040 |
| **H3:** ICA components load more on between-domain edges | **No** at α = 0.05 | Paired Wilcoxon *p* = 0.088; 20/50 components with between > within |

## 5. Discussion

### ICA vs. Dense Edges Under L2 Logistic

With **L2 logistic regression on all edges** (instead of elastic net), **ICA reached a slightly higher mean nested-CV AUC** than edges (0.838 vs. 0.823), but the **ICA vs. edges** comparison was not significant on bootstrap, label-permutation, or fold-wise Wilcoxon tests. Discrimination is therefore **statistically comparable** between the two pipelines at *N* = 363, with ICA using 50 learned features vs. 5,460 edges.

ICA **consistently** selected *k* = 50 (the **MSE grid ceiling**) in every outer fold for both H1 and H3, so reported results are **conditional on that bound**; extending *k* or using alternative ICA constraints could change conclusions.

### Between-Domain Dysconnectivity Is Statistically Significant

With validated NeuroMark 2.2 domain–subdomain labels and confound regression, H2 was now supported: between-domain edges showed significantly larger group effect sizes than within-domain edges (*p* = 0.040). This contrasts with the non-significant result obtained in earlier analyses using placeholder domain assignments, highlighting the importance of accurate functional domain labels for domain-stratified analyses. The result is consistent with the dysconnectivity hypothesis of schizophrenia, which posits that the disorder preferentially disrupts long-range inter-regional communication.

### H3 ICA Loadings Do Not Show a Global Between-Domain Advantage

With **FastICA** and *k* = 50, the **paired edge-class test was not significant** (*p* = 0.088), and mean between-domain |loading| was slightly **below** within-domain. This does **not** contradict H2 (univariate Cohen's *d* on edges): ICA seeks non-Gaussian **mixing** structure that need not align with raw group contrast maps.

### Hypothesis Domain Pairs Under ICA

The **relative ranking** of literature-defined pairs still highlights **DMN / triple-network / visual** interfaces at the top of the ICA-averaged |loading| table, while **SC-ET × TN-CE** ranked lowest **on average** across components in this run. Pair-level narrative should be treated as **descriptive** unless confirmed with inferential models that respect the exploratory, full-sample ICA fit.

### Comparison with Prior Work

Previous FBIRN FNC classification studies often report AUCs in the **0.70–0.90** range (e.g., Du et al., 2020; Salman et al., 2019). Our **nested-CV** means (~0.82–0.84) fall in that band after confound regression. **Direct numeric comparison** to older elastic-net edge models in this report is **not** appropriate because the edge classifier changed to **L2 logistic**.

## 6. Limitations and Future Directions

1. **Confound regression approach.** We used linear OLS regression to remove confound effects. Non-linear confound effects (e.g., age × diagnosis interactions) were not modeled. Additionally, medication effects were not available and could contribute to group differences.

2. **ICA search ceiling.** ICA consistently selected *k* = 50 (the upper MSE-grid bound) for both H1 and H3. Extending the range (e.g., beyond 50) may change selected dimensionality and downstream contrasts.

3. **H3 is exploratory.** FastICA was fit on **all** subjects (after confound regression), not within cross-validation folds. Fold-wise decomposition would be required for strict inferential claims about loading structure.

4. **Static FNC only.** This analysis uses static (session-level) Pearson correlations. Dynamic FNC approaches (e.g., sliding-window or time-varying connectivity) could reveal temporally specific connectivity patterns with stronger group differences.

5. **Domain-pair edge counts.** Some domain pairs (e.g., TN-SA × TN-CE with only 12 edges) have few constituent edges, which may limit the reliability of pair-specific loading estimates. The TN-CE subdomain contains only 3 ICNs in NeuroMark 2.2, constraining the resolution of central-executive interactions.

6. **Single template.** All results are conditional on the NeuroMark 2.2 105-ICN template. Replication with alternative parcellation schemes would strengthen generalizability.

## 7. Software

| Component | Version / Description |
|---|---|
| Python package | `fbirn_experiment` (custom) |
| Core dependencies | NumPy, SciPy, scikit-learn, pandas, matplotlib |
| ICN template | NeuroMark 2.2 (Iraji et al., 2023) |
| Domain labels | `data/Neuromark_fMRI_2-2_labels_final.xlsx` → `{domain_abbrev}-{subdomain_abbrev}` |
| Classification (H1) | `LogisticRegression` (L2) on edges and on ICA features |
| Decomposition (H1/H3 ICA) | `sklearn.decomposition.FastICA` (unit-variance whitening) |
| Model selection | Reconstruction MSE grid for ICA *k*; inner-CV *C* for logistic; nested stratified CV (H1) |
| H3 (default) | Same FastICA + MSE *k* selection on full-sample edges (exploratory) |
| Statistical tests | Permutation test, bootstrap, Wilcoxon signed-rank |
| Confound regression | OLS residualization (fold-wise for H1, full-sample for H2/H3) |
| Reproducibility | Fixed random seeds; single-command pipeline execution |

**Pipeline invocation:**

```bash
python -m fbirn_experiment.cli run --out results/fbirn_icn_run
```

**Key CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--outer-splits` | 5 | Number of outer CV folds |
| `--inner-splits` | 3 | Number of inner CV folds |
| `--k-min` / `--k-max` | 5 / 50 | ICA / FA component search range (H1; H3 uses same grid by default) |
| `--k-step` | 5 | Step size for component search grid |
| `--fa-criterion` | bic | BIC/AIC for **H1 FA** component count (only if `--h1-include-fa`) |
| `--h1-include-fa` | off | Add FA arm to H1 nested CV + interpretability figures |
| `--h3-fa` | off | Use **factor analysis + BIC** for H3 instead of default **ICA + MSE** |
| `--h3-no-bic` | off | Fix H3 latent dimensionality to `--h3-components` |
| `--h2-perm` | 2000 | H2 permutation iterations |
| `--auc-bootstrap` | 10000 | Bootstrap resamples for AUC comparison |
| `--auc-perm-y` | 5000 | Label-permutation iterations for AUC |
| `--confounds-csv` | auto-detect `data/FBIRN_data.csv` when present | CSV with confounding variables |
| `--confound-cols` | age sex race site hm | Columns to regress out |
| `--no-confounds` | — | Skip confound regression |

## 8. References

1. Iraji, A., et al. (2023). NeuroMark: A fully automated ICA method to identify replicable fMRI markers of brain disorders. *Human Brain Mapping*, 44(17). https://doi.org/10.1002/hbm.26472

2. Woodward, N. D., Karbasforoushan, H., & Heckers, S. (2012). Thalamocortical dysconnectivity in schizophrenia. *American Journal of Psychiatry*, 169(10), 1092–1099. https://doi.org/10.1176/appi.ajp.2012.12010056

3. Menon, V. (2011). Large-scale brain networks and psychopathology: A unifying triple network model. *Trends in Cognitive Sciences*, 15(10), 483–506. https://doi.org/10.1016/j.tics.2011.08.003

4. Palaniyappan, L., & Liddle, P. F. (2012). Does the salience network play a cardinal role in psychosis? An emerging hypothesis of insular dysfunction. *Journal of Neurology, Neurosurgery & Psychiatry*, 83(5), 558–567. https://doi.org/10.1136/jnnp-2011-301452

5. Kaufmann, T., et al. (2015). Disintegration of sensorimotor brain networks in schizophrenia. *Schizophrenia Bulletin*, 41(6), 1326–1335. https://doi.org/10.1093/schbul/sbv060

6. Marino, M., & Spironelli, C. (2022). Default mode network and auditory verbal hallucinations in schizophrenia. *Journal of Psychiatric Research*, 153, 8–16. https://doi.org/10.1016/j.jpsychires.2022.08.006

7. Manoliu, A., et al. (2014). Aberrant dependence of default mode/central executive network interactions on anterior insular salience network activity in schizophrenia. *Schizophrenia Bulletin*, 40(2), 428–437. https://doi.org/10.1093/schbul/sbt037

8. Li, S., et al. (2019). Dysconnectivity of multiple brain networks in schizophrenia: A meta-analysis of resting-state functional connectivity. *Frontiers in Psychiatry*, 10, 482. https://doi.org/10.3389/fpsyt.2019.00482

9. Giraldo-Chica, M., & Woodward, N. D. (2017). Review of thalamocortical resting-state fMRI studies in schizophrenia. *Schizophrenia Research*, 180, 58–63. https://doi.org/10.1016/j.schres.2016.08.005

10. Avram, M., et al. (2018). Aberrant striatal and thalamic tethering to auditory and somatosensory cortices in schizophrenia. *Neuropsychopharmacology*, 43(12), 2462–2471. https://doi.org/10.1038/s41386-018-0059-z

11. Ferri, J., et al. (2018). Resting-state thalamic dysconnectivity in schizophrenia and relationships with symptoms. *Scientific Reports*, 8, 3451. https://doi.org/10.1038/s41598-019-39367-z

12. Dong, D., et al. (2018). Dysfunction of large-scale brain networks in schizophrenia: A meta-analysis of resting-state functional connectivity. *Scientific Reports*, 8, 14655. https://doi.org/10.1038/srep14655

13. Sendi, M. S. E., et al. (2021). Aberrant dynamic functional connectivity of default mode network in schizophrenia and links to symptom severity. *Schizophrenia Research*, 228, 103–111. https://doi.org/10.1016/j.schres.2020.11.055

14. Du, Y., et al. (2020). NeuroMark: An automated and adaptive ICA based pipeline to identify reproducible fMRI markers of brain disorders. *NeuroImage: Clinical*, 28, 102375.

15. Salman, M. S., et al. (2019). Group ICA for identifying biomarkers in schizophrenia: 'Adaptive' networks via spatially constrained ICA show more sensitivity to group differences than spatio-temporal regression. *NeuroImage: Clinical*, 22, 101747.

16. Weber, S., et al. (2020). Auditory verbal hallucinations related alterations of default mode network connectivity. *Frontiers in Psychiatry*, 11, 227. https://doi.org/10.3389/fpsyt.2020.00227

17. Calhoun, V. D., & Sui, J. (2016). Multimodal fusion of brain imaging data: A key to finding the missing link(s) in complex mental illness. *Biological Psychiatry: Cognitive Neuroscience and Neuroimaging*, 1(3), 230–244.

---

*Report generated from run artifacts in `results/fbirn_icn_run/`. Run metadata (`run_meta.json`): 363 subjects, 157 time points, 105 ICNs, 5,460 FNC edges; confound regression (13 regressors); **H1:** edges (L2 logistic) vs **ICA** (no FA); **5-fold nested CV**; **H3:** **FastICA** with *k* chosen by **reconstruction MSE** on {5, 10, …, 50} (selected *k* = 50).*
