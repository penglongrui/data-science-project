"""
Palmer Penguins: are species or sampling island the real driver of body-size variation?
=======================================================================================

Reproducible analysis script for the CISC7204 data-science portfolio.

Research question
-----------------
Bill and flipper measurements separate the three penguin species very cleanly,
but each species was sampled on only one or two islands. This project asks:

    RQ1  How well do morphometric measurements separate the three species?
    RQ2  Is "island" a genuine driver of body size, or just a proxy for species?
    RQ3  Can a simple classifier predict species from measurements alone,
         and which measurements carry the signal?

Run:
    python analysis.py

Outputs (written next to this script):
    fig1_species_clusters.png
    fig2_island_confound.png
    fig3_model_diagnostics.png
    metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

# Everything lives in the same folder as this script, so a fresh clone runs with no
# path juggling: the data file is read from here and the figures are written back here.
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "penguins.csv"
FIG_DIR = BASE_DIR
RES_DIR = BASE_DIR

# Colour-blind-safe qualitative palette (Okabe-Ito subset).
SPECIES_COLOR = {
    "Adelie": "#0072B2",
    "Chinstrap": "#D55E00",
    "Gentoo": "#009E73",
}
ISLAND_MARKER = {"Torgersen": "o", "Biscoe": "s", "Dream": "^"}

MEASURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


# ----------------------------------------------------------------------------
# 1. Load and audit the data
# ----------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, dict]:
    """Load the raw CSV and report a data-quality summary before cleaning."""
    raw = pd.read_csv(DATA_PATH)

    audit = {
        "rows_raw": int(len(raw)),
        "missing_by_column": {k: int(v) for k, v in raw.isna().sum().items()},
        "species": raw["species"].value_counts().to_dict(),
        "islands": raw["island"].value_counts().to_dict(),
    }

    # Rows with no morphometric measurement at all cannot enter the analysis.
    clean = raw.dropna(subset=MEASURES + ["sex"]).reset_index(drop=True)
    audit["rows_clean"] = int(len(clean))
    audit["rows_dropped"] = int(len(raw) - len(clean))
    return clean, audit


# ----------------------------------------------------------------------------
# 2. Statistical helpers
# ----------------------------------------------------------------------------
def eta_squared(groups: list[np.ndarray]) -> float:
    """Proportion of total variance explained by a categorical grouping."""
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_total = ((all_values - grand_mean) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    return float(ss_between / ss_total)


def cramers_v(table: pd.DataFrame) -> float:
    """Bias-corrected Cramer's V for two categorical variables (0 = independent)."""
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2corr / min(kcorr - 1, rcorr - 1)))


# ----------------------------------------------------------------------------
# 3. Figure 1 - do the species occupy distinct measurement space?
# ----------------------------------------------------------------------------
def figure_species_clusters(df: pd.DataFrame, path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7.6, 5.6))

    for species, group in df.groupby("species"):
        for island, sub in group.groupby("island"):
            ax.scatter(
                sub["bill_length_mm"], sub["bill_depth_mm"],
                s=42, alpha=0.85, linewidths=0.6, edgecolors="white",
                color=SPECIES_COLOR[species], marker=ISLAND_MARKER[island],
                label=f"{species} · {island} (n={len(sub)})", zorder=3,
            )

    # Annotate each species centroid to make the clustering explicit.
    for species, group in df.groupby("species"):
        cx, cy = group["bill_length_mm"].median(), group["bill_depth_mm"].median()
        ax.annotate(
            species, (cx, cy), fontsize=11, fontweight="bold",
            color=SPECIES_COLOR[species], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=SPECIES_COLOR[species], lw=1.2),
            zorder=4,
        )

    ax.set_xlabel("Bill length (mm)")
    ax.set_ylabel("Bill depth (mm)")
    ax.set_title("Bill shape separates the three species\n"
                 "colour = species, marker = sampling island", fontsize=12, pad=12)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Quantitative check: how much of bill-length variance is between species?
    groups = [g["bill_length_mm"].to_numpy() for _, g in df.groupby("species")]
    f_stat, p_val = stats.f_oneway(*groups)
    return {
        "bill_length_eta_squared_by_species": round(eta_squared(groups), 3),
        "bill_length_anova_F": round(float(f_stat), 1),
        "bill_length_anova_p": float(p_val),
    }


# ----------------------------------------------------------------------------
# 4. Figure 2 - is "island" a confound or a real effect?
# ----------------------------------------------------------------------------
def figure_island_confound(df: pd.DataFrame, path: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # Left: species x island grid -> four of the nine cells are structurally empty.
    ax = axes[0]
    islands = ["Biscoe", "Dream", "Torgersen"]
    species_order = ["Adelie", "Chinstrap", "Gentoo"]
    counts = np.zeros((3, 3))
    means = np.full((3, 3), np.nan)
    for i, sp in enumerate(species_order):
        for j, isl in enumerate(islands):
            sub = df[(df["species"] == sp) & (df["island"] == isl)]
            counts[i, j] = len(sub)
            if len(sub):
                means[i, j] = sub["body_mass_g"].mean()

    shown = np.where(counts > 0, counts, np.nan)
    ax.imshow(np.ma.masked_invalid(shown), cmap="YlGnBu", aspect="auto",
              vmin=0, vmax=counts.max())
    ax.set_facecolor("#ededed")
    ax.set_xticks(range(3), islands, fontsize=10)
    ax.set_yticks(range(3), species_order, fontsize=10)
    ax.set_xlabel("Sampling island")
    for i in range(3):
        for j in range(3):
            if counts[i, j] == 0:
                ax.text(j, i, "not sampled", ha="center", va="center",
                        fontsize=9, color="#777777", style="italic")
            else:
                ax.text(j, i, f"n = {int(counts[i, j])}\n{means[i, j]:,.0f} g",
                        ha="center", va="center", fontsize=10, fontweight="bold",
                        color="white" if counts[i, j] > counts.max() * 0.55 else "#222222")
    ax.set_title("Four of nine species-island cells are empty\n"
                 "so island is a near-proxy for species", fontsize=11)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    # Right: variance in body mass explained by each grouping.
    ax = axes[1]
    eta_species = eta_squared([g["body_mass_g"].to_numpy() for _, g in df.groupby("species")])
    eta_island = eta_squared([g["body_mass_g"].to_numpy() for _, g in df.groupby("island")])
    bars = ax.bar(["Grouped by species", "Grouped by island"],
                  [eta_species, eta_island],
                  color=["#0072B2", "#999999"], width=0.55)
    for bar, val in zip(bars, [eta_species, eta_island]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share of body-mass variance explained (η²)")
    ax.set_title("Species explains far more variance\nthan sampling island", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    table = pd.crosstab(df["species"], df["island"])
    # Within-species island effect: Adelie is the only species found on all three
    # islands, so it is the one case where island can be tested independently.
    adelie = df[df["species"] == "Adelie"]
    groups = [g["body_mass_g"].to_numpy() for _, g in adelie.groupby("island")]
    eta_adelie_island = eta_squared(groups)
    p_a = float(stats.f_oneway(*groups)[1])
    return {
        "eta_squared_body_mass_by_species": round(eta_species, 3),
        "eta_squared_body_mass_by_island": round(eta_island, 3),
        "cramers_v_species_island": round(cramers_v(table), 3),
        "species_island_crosstab": table.to_dict(),
        "adelie_eta_squared_by_island": round(eta_adelie_island, 4),
        "adelie_island_anova_p": round(p_a, 4),
    }


# ----------------------------------------------------------------------------
# 5. Figure 3 - can species be predicted from measurements, and by what?
# ----------------------------------------------------------------------------
def figure_model_diagnostics(df: pd.DataFrame, path: Path) -> dict:
    X = df[MEASURES].to_numpy()
    y = df["species"].to_numpy()

    clf = RandomForestClassifier(n_estimators=500, random_state=42, min_samples_leaf=2)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_, index=MEASURES).sort_values()
    label_map = {
        "bill_length_mm": "Bill length",
        "bill_depth_mm": "Bill depth",
        "flipper_length_mm": "Flipper length",
        "body_mass_g": "Body mass",
    }
    pretty = [label_map[m] for m in importances.index]  # keep aligned after sorting

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8),
                             gridspec_kw={"width_ratios": [1, 1]})

    ax = axes[0]
    ax.barh(pretty, importances.to_numpy(), color="#0072B2", height=0.6)
    for i, v in enumerate(importances.to_numpy()):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=10)
    ax.set_xlim(0, max(importances) * 1.22)
    ax.set_xlabel("Random-forest importance")
    top_two = list(importances.index[-2:][::-1])
    ax.set_title(f"{label_map[top_two[0]]} and {label_map[top_two[1]].lower()}\n"
                 f"carry most of the species signal", fontsize=11)
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    ax = axes[1]
    order = ["Adelie", "Chinstrap", "Gentoo"]
    cm = confusion_matrix(y, y_pred, labels=order)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), order, fontsize=9)
    ax.set_yticks(range(3), order, fontsize=9)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("Actual species")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "#222222",
                    fontweight="bold")
    ax.set_title(f"5-fold cross-validated predictions\n"
                 f"accuracy = {scores.mean():.3f} ± {scores.std():.3f}", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "model": "RandomForestClassifier(n_estimators=500, min_samples_leaf=2)",
        "cv_accuracy_mean": round(float(scores.mean()), 3),
        "cv_accuracy_std": round(float(scores.std()), 3),
        "feature_importance": {k: round(float(v), 3) for k, v in importances.items()},
        "confusion_matrix": cm.tolist(),
    }


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def main() -> None:
    df, audit = load_data()
    print(f"Raw rows: {audit['rows_raw']} | analysis rows: {audit['rows_clean']} "
          f"| dropped: {audit['rows_dropped']}")
    print(f"Species: {audit['species']}")
    print(f"Islands: {audit['islands']}\n")

    metrics = {"data_audit": audit}
    metrics["species_clusters"] = figure_species_clusters(df, FIG_DIR / "fig1_species_clusters.png")
    metrics["island_confound"] = figure_island_confound(df, FIG_DIR / "fig2_island_confound.png")
    metrics["model"] = figure_model_diagnostics(df, FIG_DIR / "fig3_model_diagnostics.png")

    for section, values in metrics.items():
        print(f"[{section}]")
        for key, val in values.items():
            print(f"  {key}: {val}")
        print()

    (RES_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics written to {RES_DIR / 'metrics.json'}")
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
