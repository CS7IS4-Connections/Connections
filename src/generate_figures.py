"""
generate_figures.py
-------------------
Generates all 15 paper figures and 3 statistical tables.

Usage:
    python src/generate_figures.py \
        --input  data/results_30k/sample_30k_results_step3_entity.csv \
        --output results/figures/

Figures whose required columns are not yet present are skipped gracefully
with a console note (e.g. entity figures before entity_alignment has run,
caption-type figures before caption_classifier has run).
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

MEASURE_COLORS = {
    "tfidf_sim":   "#4C72B0",
    "jaccard_sim": "#DD8452",
    "sbert_sim":   "#55A868",
}
MEASURE_LABELS = {
    "tfidf_sim":   "TF-IDF cosine",
    "jaccard_sim": "Jaccard similarity",
    "sbert_sim":   "SBERT semantic similarity",
}
SIM_COLS = ["tfidf_sim", "jaccard_sim", "sbert_sim"]

SOURCE_COLORS = {
    "guardian": "#4C72B0",
    "bbc":      "#DD8452",
    "usatoday": "#55A868",
    "wapo":     "#C44E52",
}
SOURCE_LABELS = {
    "guardian": "The Guardian",
    "bbc":      "BBC News",
    "usatoday": "USA Today",
    "wapo":     "Washington Post",
}

CATEGORY_COLORS = {
    "politics_government":   "#4C72B0",
    "sport":                 "#DD8452",
    "business_economy":      "#55A868",
    "science_technology":    "#8172B2",
    "entertainment_culture": "#C44E52",
    "world_society":         "#64B5CD",
}
CATEGORY_LABELS = {
    "politics_government":   "Politics & Govt",
    "sport":                 "Sport",
    "business_economy":      "Business & Econ",
    "science_technology":    "Science & Tech",
    "entertainment_culture": "Entertainment",
    "world_society":         "World & Society",
}

TYPE_COLORS = {
    "Extractive":  "#4C72B0",
    "Descriptive": "#DD8452",
    "Expansive":   "#55A868",
    "Independent": "#8172B2",
}
TYPE_ORDER = ["Extractive", "Descriptive", "Expansive", "Independent"]

STRUCT_FEATURES = [
    "word_count", "avg_sent_len", "avg_word_len", "dep_depth", "clause_count",
    "ttr", "content_word_prop", "propn_prop",
    "noun_overlap", "verb_overlap", "adj_overlap", "propn_overlap",
]
STRUCT_LABELS = {
    "word_count":        "Word count",
    "avg_sent_len":      "Avg sentence length",
    "avg_word_len":      "Avg word length",
    "dep_depth":         "Dependency depth",
    "clause_count":      "Clause count",
    "ttr":               "Type–token ratio",
    "content_word_prop": "Content word prop.",
    "propn_prop":        "Proper noun prop.",
    "noun_overlap":      "Noun overlap",
    "verb_overlap":      "Verb overlap",
    "adj_overlap":       "Adjective overlap",
    "propn_overlap":     "Proper noun overlap",
}

ENTITY_OVERLAP_COLS = ["person_overlap", "gpe_overlap", "org_overlap", "date_overlap"]
ENTITY_TYPES       = ["PERSON", "GPE", "ORG", "DATE"]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.labelsize":    12,
        "axes.titlesize":    13,
        "legend.fontsize":   10,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "font.size":         10,
    })


def _has_cols(df: pd.DataFrame, cols: list[str], label: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  SKIP {label}: missing columns {missing}")
        return False
    return True


def _save(fig: plt.Figure, name: str, outdir: str, saved: list[str]) -> None:
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(path)
    print(f"  Saved {path}")


def _save_table(df: pd.DataFrame, stem: str, outdir: str, index: bool = False) -> None:
    df.to_csv(os.path.join(outdir, stem + ".csv"), index=index)
    latex = df.to_latex(index=index, escape=True)
    with open(os.path.join(outdir, stem + ".tex"), "w", encoding="utf-8") as fh:
        fh.write(latex)
    print(f"  Saved {stem}.csv / .tex")


def _ci95(series: pd.Series) -> float:
    x = series.dropna().values
    if len(x) < 2:
        return 0.0
    return float(stats.sem(x) * stats.t.ppf(0.975, len(x) - 1))


def _pearson(x: pd.Series, y: pd.Series):
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 5:
        return np.nan, np.nan
    return stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])


def _kruskal_eta2(groups: list[np.ndarray]):
    groups = [g[~np.isnan(g)] for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    H, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eta2 = (H - k + 1) / (n - k) if n > k else np.nan
    return H, p, eta2


def _stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _standardise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            std = out[c].std()
            out[c] = (out[c] - out[c].mean()) / std if std > 0 else 0.0
    return out


def _label_source(s: str) -> str:
    return SOURCE_LABELS.get(s, s)


def _label_category(c: str) -> str:
    return CATEGORY_LABELS.get(c, c)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Derive avg_word_len ad hoc if missing
    if "avg_word_len" not in df.columns:
        df["avg_word_len"] = df["char_count"] / df["word_count"].replace(0, np.nan)
        print("  Derived avg_word_len from char_count / word_count")

    # Derive is_* dummies from caption_type if present
    if "caption_type" in df.columns and "is_extractive" not in df.columns:
        for t in ["Extractive", "Descriptive", "Expansive", "Independent"]:
            df[f"is_{t.lower()}"] = (df["caption_type"] == t).astype(int)
        print("  Derived is_* dummies from caption_type")

    return df


# ─────────────────────────────────────────────────────────────
# FIG 1 — Similarity distributions (KDE)
# ─────────────────────────────────────────────────────────────

def fig01_similarity_distributions(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    if not _has_cols(df, SIM_COLS, "Fig 1"):
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for sim in SIM_COLS:
        data = df[sim].dropna()
        med  = data.median()
        color = MEASURE_COLORS[sim]
        sns.kdeplot(data, ax=ax, color=color, linewidth=2,
                    label=f"{MEASURE_LABELS[sim]} (median={med:.2f})")
        ax.axvline(med, color=color, linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Similarity score")
    ax.set_ylabel("Density")
    ax.set_title("Caption–article similarity distributions")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "fig01_similarity_distributions.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 2 — Scatter matrix (hexbin)
# ─────────────────────────────────────────────────────────────

def fig02_scatter_matrix(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    if not _has_cols(df, SIM_COLS, "Fig 2"):
        return
    plot_df = df[SIM_COLS].dropna()
    if len(plot_df) > 10_000:
        plot_df = plot_df.sample(10_000, random_state=42)

    pairs = [
        ("tfidf_sim", "jaccard_sim"),
        ("tfidf_sim", "sbert_sim"),
        ("jaccard_sim", "sbert_sim"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (xcol, ycol) in zip(axes, pairs):
        hb = ax.hexbin(plot_df[xcol], plot_df[ycol],
                       gridsize=40, cmap="Blues", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count")
        r, p = _pearson(plot_df[xcol], plot_df[ycol])
        ax.text(0.05, 0.93, f"r = {r:.2f}{_stars(p)}", transform=ax.transAxes,
                fontsize=10, va="top")
        ax.set_xlabel(MEASURE_LABELS[xcol])
        ax.set_ylabel(MEASURE_LABELS[ycol])
    fig.suptitle("Pairwise similarity measure correlations", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig02_scatter_matrix.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 3 — Structural feature correlations heatmap
# ─────────────────────────────────────────────────────────────

def fig03_structural_correlations(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    feats = [f for f in STRUCT_FEATURES if f in df.columns]
    if not _has_cols(df, feats[:3], "Fig 3"):
        return

    r_mat   = np.zeros((len(feats), 3))
    p_mat   = np.ones((len(feats), 3))
    annot   = np.empty((len(feats), 3), dtype=object)

    for i, feat in enumerate(feats):
        for j, sim in enumerate(SIM_COLS):
            r, p = _pearson(df[feat], df[sim])
            r_mat[i, j] = r if not np.isnan(r) else 0.0
            p_mat[i, j] = p if not np.isnan(p) else 1.0
            if np.isnan(r) or p >= 0.05:
                annot[i, j] = "ns"
                r_mat[i, j] = 0.0   # neutral colour for non-sig
            else:
                annot[i, j] = f"{r:.2f}{_stars(p)}"

    ylabels = [STRUCT_LABELS.get(f, f) for f in feats]
    xlabels = ["TF-IDF", "Jaccard", "SBERT"]
    r_df = pd.DataFrame(r_mat, index=ylabels, columns=xlabels)

    fig, ax = plt.subplots(figsize=(5, 7))
    sns.heatmap(
        r_df, annot=annot, fmt="",
        cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        ax=ax, linewidths=0.4,
        cbar_kws={"label": "Pearson r", "shrink": 0.6},
    )
    ax.set_title("Structural feature correlations\nwith similarity metrics")
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    fig.tight_layout()
    _save(fig, "fig03_structural_correlations.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 4 — OLS regression coefficients (structural → sbert_sim)
# ─────────────────────────────────────────────────────────────

def fig04_regression_coefs(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    import statsmodels.api as sm
    feats = [f for f in STRUCT_FEATURES if f in df.columns]
    needed = feats + ["sbert_sim"]
    if not _has_cols(df, needed, "Fig 4"):
        return

    sub = df[needed].dropna()
    std = _standardise(sub, feats)
    X   = sm.add_constant(std[feats])
    y   = std["sbert_sim"]
    res = sm.OLS(y, X).fit()

    coefs = res.params[feats]
    ci    = res.conf_int().loc[feats]
    pvals = res.pvalues[feats]

    order  = coefs.abs().sort_values(ascending=True).index
    coefs  = coefs[order]
    ci     = ci.loc[order]
    pvals  = pvals[order]
    labels = [
        (STRUCT_LABELS.get(f, f) + (" *" if pvals[f] < 0.05 else ""))
        for f in order
    ]
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in coefs]
    xerr   = np.array([coefs.values - ci[0].values, ci[1].values - coefs.values])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(coefs)), coefs.values, color=colors,
            xerr=xerr, capsize=3, error_kw={"linewidth": 1})
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Standardised coefficient (β)")
    ax.set_title("OLS predictors of SBERT similarity\n(structural features, standardised)")
    fig.tight_layout()
    _save(fig, "fig04_regression_coefs.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 5 — Caption type distribution by source
# ─────────────────────────────────────────────────────────────

def fig05_caption_type_distribution(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    if not _has_cols(df, ["source", "caption_type"], "Fig 5"):
        return

    sources = sorted(df["source"].dropna().unique(),
                     key=lambda s: SOURCE_LABELS.get(s, s))
    fig, ax = plt.subplots(figsize=(8, 4))
    x       = np.arange(len(sources))
    width   = 0.2
    offsets = np.linspace(-(len(TYPE_ORDER)-1)*width/2,
                           (len(TYPE_ORDER)-1)*width/2, len(TYPE_ORDER))

    for offset, ctype in zip(offsets, TYPE_ORDER):
        props = []
        for src in sources:
            sub = df[df["source"] == src]
            props.append((sub["caption_type"] == ctype).mean())
        bars = ax.bar(x + offset, props, width=width,
                      label=ctype, color=TYPE_COLORS[ctype])
        for bar, val in zip(bars, props):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABELS.get(s, s) for s in sources])
    ax.set_xlabel("News source")
    ax.set_ylabel("Proportion")
    ax.set_title("Caption type distribution by source")
    ax.legend(title="Caption type", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, "fig05_caption_type_distribution.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 6 — Similarity by caption type (violin + box)
# ─────────────────────────────────────────────────────────────

def fig06_similarity_by_type(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    if not _has_cols(df, ["caption_type"] + SIM_COLS, "Fig 6"):
        return

    order = [t for t in TYPE_ORDER if t in df["caption_type"].unique()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, sim in zip(axes, SIM_COLS):
        sns.violinplot(x="caption_type", y=sim, data=df, order=order,
                       palette=TYPE_COLORS, ax=ax, inner=None, alpha=0.6)
        sns.boxplot(x="caption_type", y=sim, data=df, order=order,
                    width=0.18, palette=TYPE_COLORS, ax=ax,
                    flierprops={"marker": ".", "markersize": 2},
                    boxprops={"zorder": 2})
        groups = [df[df["caption_type"] == t][sim].dropna().values for t in order]
        H, p, _ = _kruskal_eta2(groups)
        sig_str = f"H={H:.1f}, p={'<0.001' if p < 0.001 else f'{p:.3f}'}"
        print(f"  Fig 6 KW {sim}: {sig_str}")
        ax.text(0.5, 0.97, sig_str, transform=ax.transAxes,
                fontsize=8, ha="center", va="top")
        ax.set_xlabel("Caption type")
        ax.set_ylabel(MEASURE_LABELS[sim])
        ax.set_title(MEASURE_LABELS[sim])
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Similarity by caption type", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig06_similarity_by_type.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 7 — Clustering validation (silhouette + confusion matrix)
# ─────────────────────────────────────────────────────────────

def fig07_clustering_validation(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    feats = [f for f in STRUCT_FEATURES if f in df.columns]
    if len(feats) < 4:
        print("  SKIP Fig 7: too few structural features available")
        return

    has_type = "caption_type" in df.columns
    sub = df[feats + (["caption_type"] if has_type else [])].dropna()
    X   = StandardScaler().fit_transform(sub[feats])

    # Silhouette
    sil_scores = {}
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil_scores[k] = silhouette_score(X, labels)

    if has_type:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        axes = [axes]

    # Left: silhouette line chart
    ax = axes[0]
    ks = list(sil_scores.keys())
    vs = list(sil_scores.values())
    ax.plot(ks, vs, marker="o", color="#4C72B0", linewidth=2)
    ax.scatter([4], [sil_scores[4]], s=200, marker="*",
               color="#DD8452", zorder=5, label="k=4 (chosen)")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Cluster quality by k")
    ax.legend()
    ax.set_xticks(ks)

    # Right: confusion matrix vs caption_type (only if caption_type present)
    if has_type:
        km4    = KMeans(n_clusters=4, random_state=42, n_init=10)
        clust  = km4.fit_predict(X)
        types  = sub["caption_type"].values
        order  = [t for t in TYPE_ORDER if t in np.unique(types)]

        # Map cluster labels to caption types by majority vote
        type_to_idx = {t: i for i, t in enumerate(order)}
        cluster_map = {}
        for c in range(4):
            mask  = clust == c
            if mask.sum() == 0:
                cluster_map[c] = 0
                continue
            votes = [type_to_idx.get(t, 0) for t in types[mask]]
            cluster_map[c] = max(set(votes), key=votes.count)
        mapped = np.array([cluster_map[c] for c in clust])

        true_idx = np.array([type_to_idx.get(t, 0) for t in types])
        cm = confusion_matrix(true_idx, mapped,
                              labels=list(range(len(order))))
        ax2 = axes[1]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=order, yticklabels=order, ax=ax2)
        ax2.set_xlabel("KMeans cluster (mapped)")
        ax2.set_ylabel("Rule-based caption type")
        ax2.set_title("Caption type vs KMeans (k=4)")
    elif not has_type:
        print("  Fig 7: silhouette only — caption_type not yet available for confusion matrix")

    fig.suptitle("Clustering validation", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig07_clustering_validation.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 8 — Entity overlap by type (grouped bar)
# ─────────────────────────────────────────────────────────────

def fig08_entity_overlap_by_type(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    if not _has_cols(df, ENTITY_OVERLAP_COLS + ["entity_coverage"], "Fig 8"):
        return

    means = [df[c].mean() for c in ENTITY_OVERLAP_COLS]
    ses   = [df[c].sem()  for c in ENTITY_OVERLAP_COLS]
    cov_mean = df["entity_coverage"].mean()
    cov_se   = df["entity_coverage"].sem()

    x = np.arange(len(ENTITY_TYPES))
    w = 0.35
    palette = sns.color_palette("colorblind", 2)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - w/2, means, width=w, label="Overlap rate (caption→article)",
                   color=palette[0], yerr=ses, capsize=4, error_kw={"linewidth": 1})
    # Coverage as a single summary bar on top (per-type not available)
    ax.bar(x + w/2, [cov_mean]*4, width=w, label="Coverage rate (overall mean)",
           color=palette[1], alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ENTITY_TYPES)
    ax.set_xlabel("Entity type")
    ax.set_ylabel("Overlap rate (proportion)")
    ax.set_title("Entity overlap by type")
    ax.legend()
    for bar, m in zip(bars1, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005, f"{m:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig08_entity_overlap_by_type.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 9 — Entity Jaccard vs SBERT (hexbin + OLS)
# ─────────────────────────────────────────────────────────────

def fig09_entity_jaccard_vs_sbert(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    if not _has_cols(df, ["entity_jaccard", "sbert_sim"], "Fig 9"):
        return

    valid = df[["entity_jaccard", "sbert_sim"]].dropna()
    r, p  = _pearson(valid["entity_jaccard"], valid["sbert_sim"])

    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(valid["entity_jaccard"], valid["sbert_sim"],
                   gridsize=50, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count")

    # OLS line
    import statsmodels.api as sm
    X   = sm.add_constant(valid["entity_jaccard"])
    res = sm.OLS(valid["sbert_sim"], X).fit()
    xr  = np.linspace(valid["entity_jaccard"].min(), valid["entity_jaccard"].max(), 200)
    pred = res.get_prediction(sm.add_constant(xr))
    ci   = pred.conf_int()
    yhat = pred.predicted_mean
    ax.plot(xr, yhat, color="black", linewidth=1.5)
    ax.fill_between(xr, ci[:, 0], ci[:, 1], alpha=0.2, color="grey")

    p_str = "<0.001" if p < 0.001 else f"={p:.3f}"
    ax.text(0.05, 0.95, f"r = {r:.2f}, p{p_str}",
            transform=ax.transAxes, fontsize=10, va="top")
    ax.set_xlabel("Entity Jaccard similarity")
    ax.set_ylabel("SBERT semantic similarity")
    ax.set_title("Entity alignment vs semantic similarity")
    fig.tight_layout()
    _save(fig, "fig09_entity_jaccard_vs_sbert.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 10 — Predictor importance (OLS sbert ~ entity + structural)
# ─────────────────────────────────────────────────────────────

def fig10_predictor_importance(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    import statsmodels.api as sm
    needed = ["entity_jaccard", "tfidf_sim", "jaccard_sim",
              "content_word_prop", "propn_overlap", "word_count", "sbert_sim"]
    if not _has_cols(df, needed, "Fig 10"):
        return

    PREDICTORS = ["entity_jaccard", "tfidf_sim", "jaccard_sim",
                  "content_word_prop", "propn_overlap", "word_count"]
    PRED_LABELS = {
        "entity_jaccard":    "Entity Jaccard",
        "tfidf_sim":         "TF-IDF similarity",
        "jaccard_sim":       "Jaccard similarity",
        "content_word_prop": "Content word prop.",
        "propn_overlap":     "Proper noun overlap",
        "word_count":        "Word count",
    }

    sub = df[PREDICTORS + ["sbert_sim"]].dropna()
    std = _standardise(sub, PREDICTORS)
    X   = sm.add_constant(std[PREDICTORS])
    res = sm.OLS(std["sbert_sim"], X).fit()

    coefs  = res.params[PREDICTORS]
    ci     = res.conf_int().loc[PREDICTORS]
    order  = coefs.abs().sort_values(ascending=True).index
    coefs  = coefs[order]
    ci     = ci.loc[order]
    labels = [PRED_LABELS.get(f, f) for f in order]
    colors = ["#DD8452" if f == "entity_jaccard" else "#4C72B0" for f in order]
    xerr   = np.array([coefs.values - ci[0].values, ci[1].values - coefs.values])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(coefs)), coefs.values, color=colors,
            xerr=xerr, capsize=3, error_kw={"linewidth": 1})
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Standardised β")
    ax.set_title("Predictors of SBERT semantic similarity")
    amber_patch = mpatches.Patch(color="#DD8452", label="Entity Jaccard")
    blue_patch  = mpatches.Patch(color="#4C72B0", label="Other predictors")
    ax.legend(handles=[amber_patch, blue_patch], fontsize=9)
    fig.tight_layout()
    _save(fig, "fig10_predictor_importance.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 11 — Similarity by category (horizontal grouped bar, median)
# ─────────────────────────────────────────────────────────────

def fig11_similarity_by_category(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    if not _has_cols(df, ["category"] + SIM_COLS, "Fig 11"):
        return

    cats = df["category"].dropna().unique()
    sbert_medians = {c: df[df["category"] == c]["sbert_sim"].median() for c in cats}
    order = sorted(cats, key=lambda c: sbert_medians[c])   # ascending for horizontal bar

    medians = {sim: [df[df["category"] == c][sim].median() for c in order]
               for sim in SIM_COLS}

    y     = np.arange(len(order))
    h     = 0.25
    offsets = [-h, 0, h]

    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, sim in zip(offsets, SIM_COLS):
        ax.barh(y + offset, medians[sim], height=h,
                label=MEASURE_LABELS[sim], color=MEASURE_COLORS[sim])
    ax.set_yticks(y)
    ax.set_yticklabels([_label_category(c) for c in order])
    ax.set_xlabel("Median similarity score")
    ax.set_title("Caption–article similarity by news category")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "fig11_similarity_by_category.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 12 — SBERT by category (boxplot + KW + Dunn brackets)
# ─────────────────────────────────────────────────────────────

def fig12_sbert_by_category(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    if not _has_cols(df, ["category", "sbert_sim"], "Fig 12"):
        return
    try:
        import scikit_posthocs as sp
    except ImportError:
        print("  SKIP Fig 12: scikit_posthocs not installed")
        return

    cats  = df["category"].dropna().unique()
    order = sorted(cats,
                   key=lambda c: df[df["category"] == c]["sbert_sim"].median(),
                   reverse=True)

    groups  = [df[df["category"] == c]["sbert_sim"].dropna().values for c in order]
    H, p, _ = _kruskal_eta2(groups)
    p_str   = "<0.001" if p < 0.001 else f"={p:.3f}"
    print(f"  Fig 12 KW: H={H:.2f}, p{p_str}")

    dunn = sp.posthoc_dunn(df[["category", "sbert_sim"]].dropna(),
                           val_col="sbert_sim", group_col="category",
                           p_adjust="bonferroni")

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {c: CATEGORY_COLORS.get(c, "#888888") for c in order}
    sns.boxplot(x="category", y="sbert_sim", data=df, order=order,
                palette=palette, ax=ax,
                flierprops={"marker": ".", "markersize": 2},
                linewidth=0.8)
    ax.set_xticklabels([_label_category(c) for c in order], rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("SBERT semantic similarity")
    ax.set_title(f"SBERT similarity by news category\n(KW: H={H:.1f}, p{p_str})")

    # Significance brackets (max 6)
    sig_pairs = [
        (order.index(a), order.index(b))
        for a in order for b in order
        if a < b and dunn.loc[a, b] < 0.05
    ][:6]
    ymax = df["sbert_sim"].quantile(0.98)
    step = (df["sbert_sim"].max() - ymax) / max(len(sig_pairs) + 1, 1)
    for n, (i, j) in enumerate(sig_pairs):
        y  = ymax + step * (n + 1)
        ax.plot([i, i, j, j], [y - step*0.1, y, y, y - step*0.1],
                color="black", linewidth=0.8)
        ax.text((i + j) / 2, y + step*0.05, "*", ha="center", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig12_sbert_by_category.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 13 — Similarity by source (grouped bar, mean ± CI)
# ─────────────────────────────────────────────────────────────

def fig13_similarity_by_source(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    if not _has_cols(df, ["source"] + SIM_COLS, "Fig 13"):
        return

    sources = sorted(df["source"].dropna().unique(),
                     key=lambda s: SOURCE_LABELS.get(s, s))
    x     = np.arange(len(sources))
    w     = 0.25
    offsets = np.linspace(-(len(SIM_COLS)-1)*w/2,
                           (len(SIM_COLS)-1)*w/2, len(SIM_COLS))

    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, sim in zip(offsets, SIM_COLS):
        means = [df[df["source"] == s][sim].mean() for s in sources]
        cis   = [_ci95(df[df["source"] == s][sim]) for s in sources]
        ax.bar(x + offset, means, width=w,
               label=MEASURE_LABELS[sim], color=MEASURE_COLORS[sim],
               yerr=cis, capsize=4, error_kw={"linewidth": 1})

        # KW p-value annotation
        groups  = [df[df["source"] == s][sim].dropna().values for s in sources]
        H, p, _ = _kruskal_eta2(groups)
        if not np.isnan(p):
            p_str = f"p{'<0.001' if p < 0.001 else f'={p:.3f}'}"
            print(f"  Fig 13 KW {sim}: H={H:.1f}, {p_str}")

    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABELS.get(s, s) for s in sources])
    ax.set_xlabel("News source")
    ax.set_ylabel("Mean similarity score")
    ax.set_title("Caption–article similarity by news source")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "fig13_similarity_by_source.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 14 — Entity alignment by source (bar chart)
# ─────────────────────────────────────────────────────────────

def fig14_entity_overlap_heatmap(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    if not _has_cols(df, ["source", "entity_jaccard"], "Fig 14"):
        return

    sources = sorted(df["source"].dropna().unique(),
                     key=lambda s: SOURCE_LABELS.get(s, s))
    means = [df[df["source"] == s]["entity_jaccard"].mean() for s in sources]
    ses   = [df[df["source"] == s]["entity_jaccard"].sem()  for s in sources]
    colors = [SOURCE_COLORS.get(s, "#888888") for s in sources]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(range(len(sources)), means, xerr=ses,
                   color=colors, capsize=4, error_kw={"linewidth": 1})
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([SOURCE_LABELS.get(s, s) for s in sources])
    ax.set_xlabel("Mean entity Jaccard similarity")
    ax.set_title("Entity alignment by news source")
    for bar, m in zip(bars, means):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{m:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig14_entity_overlap_heatmap.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# FIG 15 — Source × Category heatmap (median SBERT)
# ─────────────────────────────────────────────────────────────

def fig15_source_category_heatmap(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    if not _has_cols(df, ["source", "category", "sbert_sim"], "Fig 15"):
        return

    pivot = df.pivot_table(
        values="sbert_sim", index="source", columns="category", aggfunc="median"
    )
    pivot.index   = [SOURCE_LABELS.get(s, s)   for s in pivot.index]
    pivot.columns = [_label_category(c)         for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                ax=ax, cbar_kws={"label": "Median SBERT similarity"})
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    ax.set_title("Median SBERT similarity by source and category")
    fig.tight_layout()
    _save(fig, "fig15_source_category_heatmap.png", outdir, saved)


# ─────────────────────────────────────────────────────────────
# TABLE 1 — Descriptive statistics
# ─────────────────────────────────────────────────────────────

def table1_descriptive_stats(df: pd.DataFrame, outdir: str) -> None:
    rows = []
    groups = list(df["source"].dropna().unique()) + ["Overall"]
    for grp in groups:
        sub = df if grp == "Overall" else df[df["source"] == grp]
        row = {"Source": SOURCE_LABELS.get(grp, grp)}
        for col, label in [("tfidf_sim", "TF-IDF"), ("jaccard_sim", "Jaccard"),
                            ("sbert_sim", "SBERT"),  ("word_count", "Word count")]:
            if col not in sub.columns:
                row[f"{label} mean"] = row[f"{label} median"] = row[f"{label} SD"] = "—"
            else:
                d = sub[col].dropna()
                row[f"{label} mean"]   = f"{d.mean():.3f}"
                row[f"{label} median"] = f"{d.median():.3f}"
                row[f"{label} SD"]     = f"{d.std():.3f}"
        rows.append(row)
    _save_table(pd.DataFrame(rows), "table1_descriptive_stats", outdir)


# ─────────────────────────────────────────────────────────────
# TABLE 2 — Kruskal-Wallis statistical tests
# ─────────────────────────────────────────────────────────────

def table2_statistical_tests(df: pd.DataFrame, outdir: str) -> None:
    rows = []
    comparisons = []
    for grp_col, grp_label in [("caption_type", "Caption type"),
                                ("category",     "News category"),
                                ("source",       "News source")]:
        if grp_col not in df.columns:
            continue
        for sim in SIM_COLS:
            groups_vals = df[grp_col].dropna().unique()
            arrays = [df[df[grp_col] == g][sim].dropna().values for g in groups_vals]
            H, p, eta2 = _kruskal_eta2(arrays)
            rows.append({
                "Comparison":  grp_label,
                "Measure":     MEASURE_LABELS[sim],
                "H statistic": f"{H:.2f}" if not np.isnan(H) else "—",
                "p-value":     f"{p:.4f}" if not np.isnan(p) else "—",
                "eta_squared": f"{eta2:.4f}" if not np.isnan(eta2) else "—",
                "Significant": "Yes" if (not np.isnan(p) and p < 0.05) else "No",
            })
    if rows:
        _save_table(pd.DataFrame(rows), "table2_statistical_tests", outdir)


# ─────────────────────────────────────────────────────────────
# TABLE 3 — Full OLS regression
# ─────────────────────────────────────────────────────────────

def table3_regression(df: pd.DataFrame, outdir: str) -> None:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    struct = [f for f in STRUCT_FEATURES if f in df.columns]
    entity = ["entity_jaccard"] if "entity_jaccard" in df.columns else []
    type_dummies = [c for c in ["is_extractive", "is_descriptive", "is_expansive"]
                    if c in df.columns]
    needed = struct + entity + type_dummies + ["sbert_sim", "source", "category"]
    if not _has_cols(df, struct + ["sbert_sim"], "Table 3"):
        return

    sub = df[needed].dropna()

    # Source and category dummies
    src_dummies = pd.get_dummies(sub["source"],   prefix="src",  drop_first=True)
    cat_dummies = pd.get_dummies(sub["category"], prefix="cat",  drop_first=True)
    pred_cols   = struct + entity + type_dummies + list(src_dummies.columns) + list(cat_dummies.columns)

    feat_df = pd.concat([sub[struct + entity + type_dummies].reset_index(drop=True),
                          src_dummies.reset_index(drop=True),
                          cat_dummies.reset_index(drop=True)], axis=1)
    std_df  = _standardise(feat_df, pred_cols)
    X       = sm.add_constant(std_df[pred_cols])
    y       = _standardise(sub[["sbert_sim"]], ["sbert_sim"])["sbert_sim"].values

    res = sm.OLS(y, X).fit()

    # VIF
    vif_vals = {}
    X_no_const = std_df[pred_cols].copy()
    for i, col in enumerate(pred_cols):
        try:
            vif_vals[col] = variance_inflation_factor(X_no_const.values, i)
        except Exception:
            vif_vals[col] = np.nan

    rows = []
    for pred in pred_cols:
        if pred not in res.params:
            continue
        vif = vif_vals.get(pred, np.nan)
        rows.append({
            "Predictor":   pred,
            "β":           f"{res.params[pred]:.3f}",
            "SE":          f"{res.bse[pred]:.3f}",
            "t":           f"{res.tvalues[pred]:.2f}",
            "p-value":     f"{res.pvalues[pred]:.4f}",
            "VIF":         f"{vif:.2f}" if not np.isnan(vif) else "—",
            "VIF flagged": "Yes" if (not np.isnan(vif) and vif > 5) else "No",
        })
    _save_table(pd.DataFrame(rows), "table3_regression", outdir)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures and tables")
    parser.add_argument("--input",  required=True, help="Pipeline output CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    df = load_data(args.input)
    os.makedirs(args.output, exist_ok=True)
    _setup_style()

    print(f"\nLoaded {len(df):,} rows from {args.input}")
    print(f"Columns: {list(df.columns)}\n")

    saved: list[str] = []

    print("--- Descriptive ---")
    fig01_similarity_distributions(df, args.output, saved)
    fig02_scatter_matrix(df, args.output, saved)

    print("\n--- RQ1: Structural ---")
    fig03_structural_correlations(df, args.output, saved)
    fig04_regression_coefs(df, args.output, saved)

    print("\n--- RQ2: Caption type ---")
    fig05_caption_type_distribution(df, args.output, saved)
    fig06_similarity_by_type(df, args.output, saved)
    fig07_clustering_validation(df, args.output, saved)

    print("\n--- RQ3: Entity ---")
    fig08_entity_overlap_by_type(df, args.output, saved)
    fig09_entity_jaccard_vs_sbert(df, args.output, saved)
    fig10_predictor_importance(df, args.output, saved)

    print("\n--- RQ4: Category ---")
    fig11_similarity_by_category(df, args.output, saved)
    fig12_sbert_by_category(df, args.output, saved)

    print("\n--- RQ5: Source ---")
    fig13_similarity_by_source(df, args.output, saved)
    fig14_entity_overlap_heatmap(df, args.output, saved)
    fig15_source_category_heatmap(df, args.output, saved)

    print("\n--- Tables ---")
    table1_descriptive_stats(df, args.output)
    table2_statistical_tests(df, args.output)
    table3_regression(df, args.output)

    print(f"\nDone: {len(saved)} figures saved to {args.output}")


if __name__ == "__main__":
    main()
