# Does sampling island drive penguin body size, or is it just a proxy for species?

A small, fully reproducible analysis of the **Palmer Penguins** dataset that separates a real
biological signal (species) from a structural artefact of how the data were collected (island).

**Author:** Peng LongRui · CISC7204 Data Visualisation, University of Macau
**Status:** Completed

---

## 1. Problem and research question

The Palmer Penguins dataset is a standard teaching example, and it is usually presented as
"three clearly separated penguin species". The three species, however, were **not sampled
evenly across the three islands**: Gentoo penguins were recorded only on Biscoe, Chinstrap
penguins only on Dream. This creates a collinearity problem — a large part of the apparent
"island effect" may simply be the species effect in disguise.

The research question is therefore:

> **How much of the variation in penguin body size is explained by species, and how much is
> genuinely explained by the sampling island once species is held constant?**

Three sub-questions guide the analysis:

| # | Sub-question | How it is answered |
|---|---|---|
| RQ1 | Can the three species be separated from measurements alone? | Scatter plot of bill shape + one-way ANOVA / η² |
| RQ2 | Is island a driver of body size, or a proxy for species? | Species × island cross-tabulation, Cramér's V, η² comparison, within-species test |
| RQ3 | Which measurements carry the species signal? | Random forest feature importance + 5-fold cross-validated confusion matrix |

## 2. Dataset: source and usage permissions

| Item | Detail |
|---|---|
| Dataset | Palmer Archipelago (Antarctica) penguin data — `penguins.csv` |
| Original collectors | Dr. Kristen Gorman (Palmer Station LTER) and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network |
| Compiled by | Allison Horst, Alison Hill and Kristen Gorman |
| Source | <https://github.com/allisonhorst/palmerpenguins> (`inst/extdata/penguins.csv`) |
| Licence | **CC0 1.0 Universal (public domain dedication)** — free to use, modify and redistribute for any purpose, including commercial use, without permission |
| Citation requested | Horst AM, Hill AP, Gorman KB (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. R package version 0.1.0. <https://allisonhorst.github.io/palmerpenguins/> |
| Personal data | None. The records describe animals, not people; there is no identifier that could re-identify an individual |
| Copy in this repo | `penguins.csv` (344 rows × 8 columns, 15 KB) is redistributed under the same CC0 terms so that the analysis runs offline |

**Sensitive information:** the repository contains no credentials, API keys, personal data or
restricted data.

## 3. Methods and tools

**Tools.** Python 3.13 · pandas (loading, cleaning, cross-tabulation) · NumPy (variance
decomposition) · SciPy (one-way ANOVA, χ²) · scikit-learn (random forest, stratified 5-fold
cross-validation) · Matplotlib (all figures). The whole pipeline is a single script,
`analysis.py`, so anyone can re-run it end to end.

**Method.**

1. **Data audit.** Count missing values per column before touching anything; drop the 11 rows
   that lack a measurement or a recorded sex, leaving 333 complete records (Adelie 146,
   Chinstrap 68, Gentoo 119). Documenting the loss (3.2 %) is part of the result, not a
   detail to hide.
2. **Visual separation (RQ1).** Scatter plot of bill length against bill depth, encoding
   species as colour and island as marker shape, so both variables can be read at once.
   Support the picture with a one-way ANOVA and η² (share of variance explained).
3. **Confounding check (RQ2).** Cross-tabulate species × island, quantify association with
   bias-corrected Cramér's V, and compare η² for body mass grouped by species versus grouped
   by island. The decisive test is *within species*: Adelie is the only species present on all
   three islands, so only Adelie allows island to be varied while species is held constant.
4. **Predictive check (RQ3).** Random forest (500 trees, `min_samples_leaf=2`) trained on the
   four morphometric measurements, evaluated with stratified 5-fold cross-validation;
   report feature importance and the out-of-fold confusion matrix.

**Statistics used:** η² (variance explained), Cramér's V (association between two categorical
variables), one-way ANOVA F-test, and cross-validated accuracy. No p-hacking, no test set
tuning — the model is never fitted on the data used to score it.

## 4. Key findings

1. **The species really are separable — bill shape alone explains 71 % of bill-length
   variance** (one-way ANOVA, F = 397, p < 0.001). Bill length plus flipper length carry
   almost all the signal (importance 0.42 and 0.34); body mass adds little (0.08).
2. **A random forest predicts species from four measurements with 97.0 % ± 2.3 % cross-validated
   accuracy** (10 out-of-fold errors out of 333). Nine of those ten errors are Adelie ↔ Chinstrap
   confusion — the two species that share the island of Dream — plus one Chinstrap predicted as
   Gentoo.
3. **"Island" is largely a proxy for species, not an independent driver.** Species explains
   67 % of body-mass variance but island only 39 %; four of the nine species × island cells
   are structurally empty (no Gentoo outside Biscoe, no Chinstrap outside Dream).
4. **Once species is held constant, island has essentially no effect.** For Adelie — the only
   species sampled on all three islands — mean body mass is 3,710 g / 3,701 g / 3,709 g on
   Biscoe / Dream / Torgersen, i.e. island explains **0.01 %** of the variance (p = 0.995).
   The "island effect" in the full dataset is therefore the Gentoo effect in disguise.

### One limitation

The dataset is **observational and spatially unreplicated**: each species occurs on at most two
of the three islands, so species and island cannot be fully disentangled for Chinstrap and
Gentoo — the 0.01 % figure above is only available for Adelie. The data also cover three
summers (2007–2009) at a single archipelago, so the findings describe *this* colony and should
not be generalised to a claim that island never matters, nor to any causal statement about
why Gentoo penguins are heavier.

## 5. Visualisations

| Figure | What it shows |
|---|---|
| [`fig1_species_clusters.png`](fig1_species_clusters.png) | Bill length vs bill depth; colour = species, marker = island |
| [`fig2_island_confound.png`](fig2_island_confound.png) | Species × island grid (empty cells) and η² by grouping |
| [`fig3_model_diagnostics.png`](fig3_model_diagnostics.png) | Random-forest feature importance and cross-validated confusion matrix |

![Bill shape separates the three species](fig1_species_clusters.png)

![Island is a proxy for species](fig2_island_confound.png)

## 6. How to run the analysis

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/data-science-project.git
cd data-science-project

# 2. Create an isolated environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the pinned dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python analysis.py
```

Expected console output (abridged):

```
Raw rows: 344 | analysis rows: 333 | dropped: 11
...
eta_squared_body_mass_by_species: 0.674
eta_squared_body_mass_by_island: 0.389
adelie_eta_squared_by_island: 0.0001
adelie_island_anova_p: 0.9952
cv_accuracy_mean: 0.97
```

Outputs are written next to the script: three PNG files at 200 dpi
(`fig1_species_clusters.png`, `fig2_island_confound.png`, `fig3_model_diagnostics.png`) and
`metrics.json`, which holds every number quoted above. The run is deterministic: all random
seeds are fixed at 42, so a fresh clone reproduces the figures and metrics exactly.

To refresh the data from the upstream source instead of using the bundled copy:

```bash
curl -o penguins.csv \
  https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv
```

## 7. Repository structure

The repository is deliberately flat: one script, one data file, three figures and one metrics
file, all in the root folder. Nothing is buried in sub-folders, and every path the code writes
matches the path documented here.

```
data-science-project/
├── analysis.py                    # the entire, reproducible pipeline
├── penguins.csv                   # CC0 dataset copy (no download needed to reproduce)
├── fig1_species_clusters.png      # generated: bill shape by species and island
├── fig2_island_confound.png       # generated: species x island grid + eta-squared
├── fig3_model_diagnostics.png     # generated: feature importance + confusion matrix
├── metrics.json                   # generated: every number quoted in this README
├── requirements.txt               # runtime dependencies
├── LICENSE                        # MIT for the code; data stays CC0
└── README.md                      # this document
```

## 8. Licence

- **Code** (`analysis.py`): MIT — see [LICENSE](LICENSE).
- **Data** (`penguins.csv`): CC0 1.0 public domain, courtesy of the Palmer Station LTER
  and the `palmerpenguins` authors.
