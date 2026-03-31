"""
visualizations.py
-----------------
Generate all figures and statistical tables for the caption-article
similarity paper.

Usage:
    python src/visualizations.py \\
        --input  results/sample_results.csv \\
        --output results/figures/
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
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRUCT_FEATURES = [
    "token_count", "word_count", "char_count", "sent_count",
    "avg_sent_len", "dep_depth", "clause_count",
    "ttr", "content_word_prop", "propn_prop",
    "noun_overlap", "verb_overlap", "adj_overlap", "propn_overlap",
]

FEAT_LABELS = {
    "token_count":       "token count",
    "word_count":        "word count",
    "char_count":        "char count",
    "sent_count":        "sent count",
    "avg_sent_len":      "avg sent len",
    "dep_depth":         "dep depth",
    "clause_count":      "clause count",
    "ttr":               "TTR",
    "content_word_prop": "content word prop",
    "propn_prop":        "propn prop",
    "noun_overlap":      "noun overlap",
    "verb_overlap":      "verb overlap",
    "adj_overlap":       "adj overlap",
    "propn_overlap":     "propn overlap",
}

SIM_COLS   = ["tfidf_sim", "jaccard_sim", "sbert_sim"]
SIM_LABELS = {"tfidf_sim": "TF-IDF", "jaccard_sim": "Jaccard", "sbert_sim": "SBERT"}

ENTITY_TYPES = ["PERSON", "GPE", "ORG", "DATE"]
OVERLAP_COLS = ["person_overlap", "gpe_overlap", "org_overlap", "date_overlap"]

FONT_LABEL = 10
FONT_TICK  = 8

PALETTE = sns.color_palette("colorblind")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    plt.rcParams.update({
        "font.size":         FONT_TICK,
        "axes.labelsize":    FONT_LABEL,
        "xtick.labelsize":   FONT_TICK,
        "ytick.labelsize":   FONT_TICK,
        "legend.fontsize":   FONT_TICK,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _ci95(series: pd.Series) -> float:
    x = series.dropna()
    n = len(x)
    if n < 2:
        return 0.0
    return float(stats.sem(x) * stats.t.ppf(0.975, n - 1))


def _save_png(fig: plt.Figure, name: str, outdir: str, saved: list[str]) -> None:
    path = os.path.join(outdir, name + ".png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)


def _save_table(df: pd.DataFrame, stem: str, outdir: str, index: bool = False) -> None:
    df.to_csv(os.path.join(outdir, stem + ".csv"), index=index)
    latex = df.to_latex(index=index, escape=True)
    with open(os.path.join(outdir, stem + ".tex"), "w") as fh:
        fh.write(latex)


def _kruskal_eta2(groups: list[np.ndarray]) -> tuple[float, float, float]:
    """Returns (H, p, eta_squared)."""
    groups = [g[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    res = stats.kruskal(*groups)
    H, p = res.statistic, res.pvalue
    n = sum(len(g) for g in groups)
    k = len(groups)
    eta2 = (H - k + 1) / (n - k) if n > k else np.nan
    return H, p, eta2


def _three_panel_boxplot(
    df: pd.DataFrame,
    group_col: str,
    outdir: str,
    fname: str,
    saved: list[str],
    rotate_x: int = 0,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7, 3.5), sharey=False)
    order = sorted(df[group_col].dropna().unique())
    for ax, sim in zip(axes, SIM_COLS):
        sns.boxplot(
            x=group_col, y=sim, data=df, order=order,
            palette=PALETTE, ax=ax,
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
            linewidth=0.8,
        )
        sns.stripplot(
            x=group_col, y=sim, data=df, order=order,
            ax=ax, alpha=0.3, size=2, color="0.4", jitter=True,
        )
        ax.set_xlabel(group_col.replace("_", " "), fontsize=FONT_LABEL)
        ax.set_ylabel(SIM_LABELS[sim] + " similarity", fontsize=FONT_LABEL)
        if rotate_x:
            ax.tick_params(axis="x", rotation=rotate_x)
    fig.tight_layout()
    _save_png(fig, fname, outdir, saved)


def _mean_sd_table(
    df: pd.DataFrame, group_col: str
) -> pd.DataFrame:
    rows = []
    groups = df[group_col].dropna().unique()
    for g in sorted(groups):
        sub = df[df[group_col] == g]
        row = {group_col: g}
        for sim in SIM_COLS:
            m = sub[sim].mean()
            s = sub[sim].std()
            row[SIM_LABELS[sim]] = f"{m:.3f} ± {s:.3f}"
        rows.append(row)
    # Kruskal-Wallis footer
    kw_row = {group_col: "Kruskal-Wallis"}
    for sim in SIM_COLS:
        grp_arrays = [df[df[group_col] == g][sim].dropna().values for g in sorted(groups)]
        H, p, _ = _kruskal_eta2(grp_arrays)
        kw_row[SIM_LABELS[sim]] = f"H={H:.2f}, p={p:.3f}" if not np.isnan(H) else "—"
    rows.append(kw_row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Descriptive figures
# ---------------------------------------------------------------------------

def fig_caption_length(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    data = df["word_count"].dropna()
    ax.hist(data, bins=40, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    med = data.median()
    ax.axvline(med, color="black", linestyle="--", linewidth=1)
    ax.text(med + 0.5, ax.get_ylim()[1] * 0.88, f"median = {med:.0f}",
            fontsize=FONT_TICK, va="top")
    ax.set_xlabel("Caption word count")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    _save_png(fig, "caption_length_distribution", outdir, saved)


def fig_similarity_distributions(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7, 3))
    for i, (ax, sim) in enumerate(zip(axes, SIM_COLS)):
        data = df[sim].dropna()
        ax.hist(data, bins=40, color=PALETTE[1], edgecolor="white", linewidth=0.5)
        mean_v = data.mean()
        med_v  = data.median()
        ax.axvline(mean_v, color="black", linestyle="-",  linewidth=1, label=f"mean={mean_v:.2f}")
        ax.axvline(med_v,  color="black", linestyle="--", linewidth=1, label=f"med={med_v:.2f}")
        ax.set_xlabel(SIM_LABELS[sim] + " similarity")
        ax.set_ylabel("Frequency" if i == 0 else "")
        ax.legend(fontsize=6)
    fig.tight_layout()
    _save_png(fig, "similarity_score_distributions", outdir, saved)


# ---------------------------------------------------------------------------
# RQ1 — Structural features
# ---------------------------------------------------------------------------

def _compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in STRUCT_FEATURES:
        if feat not in df.columns:
            continue
        row: dict = {"feature": feat, "feature_label": FEAT_LABELS.get(feat, feat)}
        for sim in SIM_COLS:
            valid = df[[feat, sim]].dropna()
            if len(valid) < 5:
                row[f"r_{sim}"] = np.nan
                row[f"p_{sim}"] = np.nan
            else:
                r, p = stats.pearsonr(valid[feat], valid[sim])
                row[f"r_{sim}"] = r
                row[f"p_{sim}"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def fig_rq1_heatmap(corr_df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    feat_labels = corr_df["feature_label"].tolist()
    r_mat = np.column_stack([corr_df[f"r_{s}"].values for s in SIM_COLS]).astype(float)
    p_mat = np.column_stack([corr_df[f"p_{s}"].values for s in SIM_COLS]).astype(float)

    col_labels = [SIM_LABELS[s] for s in SIM_COLS]
    r_df = pd.DataFrame(r_mat, index=feat_labels, columns=col_labels)

    annot = np.empty_like(r_mat, dtype=object)
    for i in range(r_mat.shape[0]):
        for j in range(r_mat.shape[1]):
            rv, pv = r_mat[i, j], p_mat[i, j]
            if np.isnan(rv):
                annot[i, j] = ""
            elif not np.isnan(pv) and pv < 0.05:
                annot[i, j] = f"{rv:.2f}*"
            else:
                annot[i, j] = f"{rv:.2f}"

    fig, ax = plt.subplots(figsize=(4, 6))
    sns.heatmap(
        r_df, annot=annot, fmt="",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        ax=ax, linewidths=0.4,
        cbar_kws={"label": "Pearson r", "shrink": 0.6},
    )
    ax.tick_params(axis="y", rotation=0, labelsize=FONT_TICK)
    ax.tick_params(axis="x", labelsize=FONT_LABEL)
    fig.tight_layout()
    _save_png(fig, "rq1_correlation_heatmap", outdir, saved)


def fig_rq1_top_scatter(
    df: pd.DataFrame, corr_df: pd.DataFrame, outdir: str, saved: list[str]
) -> None:
    candidates = []
    for _, row in corr_df.iterrows():
        for sim in SIM_COLS:
            r = row[f"r_{sim}"]
            p = row[f"p_{sim}"]
            if not np.isnan(r) and not np.isnan(p) and p < 0.05:
                candidates.append((abs(r), row["feature"], sim, r))
    candidates.sort(reverse=True)
    top = candidates[:3]

    if not top:
        print("  [RQ1] No significant correlations found; skipping top scatter.")
        return

    fig, axes = plt.subplots(1, len(top), figsize=(7, 3))
    if len(top) == 1:
        axes = [axes]
    for ax, (_, feat, sim, r_val) in zip(axes, top):
        valid = df[[feat, sim]].dropna()
        sns.regplot(
            x=feat, y=sim, data=valid, ax=ax, ci=95,
            scatter_kws={"alpha": 0.25, "s": 8, "color": PALETTE[2], "rasterized": True},
            line_kws={"color": "black", "linewidth": 1},
        )
        ax.set_xlabel(FEAT_LABELS.get(feat, feat))
        ax.set_ylabel(SIM_LABELS[sim] + " sim")
        ax.text(0.05, 0.95, f"r = {r_val:.2f}*", transform=ax.transAxes,
                fontsize=FONT_TICK, va="top")
    fig.tight_layout()
    _save_png(fig, "rq1_top_scatter", outdir, saved)


def table_rq1(corr_df: pd.DataFrame, outdir: str) -> None:
    tbl = corr_df.copy()
    tbl["abs_r"] = tbl[f"r_{SIM_COLS[-1]}"].abs()
    tbl = tbl.sort_values("abs_r", ascending=False).drop(columns="abs_r")

    out = pd.DataFrame()
    out["feature"] = tbl["feature_label"]
    for sim in SIM_COLS:
        label = SIM_LABELS[sim]
        out[f"r ({label})"] = tbl[f"r_{sim}"].map(
            lambda x: f"{x:.3f}" if not pd.isna(x) else "—"
        )
        out[f"p ({label})"] = tbl[f"p_{sim}"].map(
            lambda x: (f"{x:.3f}*" if x < 0.05 else f"{x:.3f}") if not pd.isna(x) else "—"
        )
    _save_table(out, "table_rq1_correlations", outdir)


# ---------------------------------------------------------------------------
# RQ2 — Caption function
# ---------------------------------------------------------------------------

def fig_rq2(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    _three_panel_boxplot(df, "caption_type", outdir, "rq2_similarity_by_type", saved)


def table_rq2(df: pd.DataFrame, outdir: str) -> None:
    tbl = _mean_sd_table(df, "caption_type")
    _save_table(tbl, "table_rq2_similarity_by_type", outdir)


# ---------------------------------------------------------------------------
# RQ3 — Entity alignment
# ---------------------------------------------------------------------------

def fig_rq3_entity_bar(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    means = [df[col].mean() for col in OVERLAP_COLS]
    cis   = [_ci95(df[col]) for col in OVERLAP_COLS]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(4)
    bars = ax.bar(x, means, yerr=cis, color=PALETTE[:4], capsize=5,
                  width=0.55, error_kw={"linewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(ENTITY_TYPES, fontsize=FONT_LABEL)
    ax.set_ylabel("Mean overlap rate")
    ymax = max(m + c for m, c in zip(means, cis)) if means else 1
    ax.set_ylim(0, ymax * 1.25)
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.03,
            f"{m:.3f}", ha="center", va="bottom", fontsize=FONT_TICK,
        )
    fig.tight_layout()
    _save_png(fig, "rq3_entity_overlap_by_type", outdir, saved)


def fig_rq3_entity_scatter(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    valid = df[["entity_jaccard", "sbert_sim", "source"]].dropna()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sources = sorted(valid["source"].unique())
    for i, src in enumerate(sources):
        sub = valid[valid["source"] == src]
        ax.scatter(sub["entity_jaccard"], sub["sbert_sim"],
                   label=src, alpha=0.4, s=10, color=PALETTE[i % len(PALETTE)],
                   rasterized=True)
    # Regression line across all
    sns.regplot(
        x="entity_jaccard", y="sbert_sim", data=valid, ax=ax, ci=95,
        scatter=False,
        line_kws={"color": "black", "linewidth": 1},
    )
    r, p = stats.pearsonr(valid["entity_jaccard"], valid["sbert_sim"])
    ax.text(0.05, 0.95, f"r = {r:.2f}{'*' if p < 0.05 else ''}",
            transform=ax.transAxes, fontsize=FONT_TICK, va="top")
    ax.set_xlabel("entity Jaccard")
    ax.set_ylabel("SBERT similarity")
    ax.legend(fontsize=6, markerscale=1.5)
    fig.tight_layout()
    _save_png(fig, "rq3_entity_jaccard_vs_sbert", outdir, saved)


def table_rq3(df: pd.DataFrame, outdir: str) -> None:
    rows = []
    for etype, col in zip(ENTITY_TYPES, OVERLAP_COLS):
        rows.append({
            "entity type":   etype,
            "mean overlap":  f"{df[col].mean():.3f}",
            "mean coverage": "—",
            "mean jaccard":  "—",
        })
    rows.append({
        "entity type":   "Overall",
        "mean overlap":  f"{df[OVERLAP_COLS].mean(axis=1).mean():.3f}",
        "mean coverage": f"{df['entity_coverage'].mean():.3f}",
        "mean jaccard":  f"{df['entity_jaccard'].mean():.3f}",
    })
    _save_table(pd.DataFrame(rows), "table_rq3_entity_metrics", outdir)


# ---------------------------------------------------------------------------
# RQ4 — News category
# ---------------------------------------------------------------------------

def fig_rq4(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    _three_panel_boxplot(
        df, "category", outdir, "rq4_similarity_by_category", saved, rotate_x=30
    )


def table_rq4(df: pd.DataFrame, outdir: str) -> None:
    tbl = _mean_sd_table(df, "category")
    _save_table(tbl, "table_rq4_similarity_by_category", outdir)


# ---------------------------------------------------------------------------
# RQ5 — Cross-source
# ---------------------------------------------------------------------------

def fig_rq5_by_source(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    _three_panel_boxplot(df, "source", outdir, "rq5_similarity_by_source", saved)


def fig_rq5_entity_bar(df: pd.DataFrame, outdir: str, saved: list[str]) -> None:
    sources = sorted(df["source"].dropna().unique())
    means   = [df[df["source"] == s]["entity_jaccard"].mean() for s in sources]
    cis     = [_ci95(df[df["source"] == s]["entity_jaccard"]) for s in sources]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x    = np.arange(len(sources))
    bars = ax.bar(x, means, yerr=cis, color=PALETTE[:len(sources)], capsize=5,
                  width=0.55, error_kw={"linewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(sources, fontsize=FONT_LABEL)
    ax.set_ylabel("Mean entity Jaccard")
    ymax = max(m + c for m, c in zip(means, cis)) if means else 1
    ax.set_ylim(0, ymax * 1.25)
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.03,
            f"{m:.3f}", ha="center", va="bottom", fontsize=FONT_TICK,
        )
    fig.tight_layout()
    _save_png(fig, "rq5_entity_jaccard_by_source", outdir, saved)


def table_rq5(df: pd.DataFrame, outdir: str) -> None:
    tbl = _mean_sd_table(df, "source")
    _save_table(tbl, "table_rq5_similarity_by_source", outdir)


def table_rq5_dunn(df: pd.DataFrame, outdir: str) -> None:
    try:
        import scikit_posthocs as sp
    except ImportError:
        print("  [RQ5] scikit-posthocs not installed; skipping Dunn post-hoc table.")
        return

    valid = df[["source", "sbert_sim"]].dropna()
    if valid["source"].nunique() < 2:
        print("  [RQ5] Fewer than 2 sources; skipping Dunn post-hoc.")
        return

    dunn = sp.posthoc_dunn(valid, val_col="sbert_sim", group_col="source",
                           p_adjust="bonferroni")
    dunn_fmt = dunn.map(lambda x: f"{x:.3f}")
    _save_table(dunn_fmt, "table_rq5_dunn_posthoc", outdir, index=True)


# ---------------------------------------------------------------------------
# Dataset statistics table
# ---------------------------------------------------------------------------

def table_dataset_stats(df: pd.DataFrame, outdir: str) -> None:
    rows = []
    for src in sorted(df["source"].dropna().unique()):
        sub = df[df["source"] == src]
        has_entity = (
            (sub[OVERLAP_COLS] > 0).any(axis=1) | (sub["entity_jaccard"] > 0)
        )
        rows.append({
            "source":                  src,
            "rows":                    len(sub),
            "avg caption words":       f"{sub['word_count'].mean():.1f}",
            "avg article words":       f"{sub['article_lead'].dropna().str.split().str.len().mean():.1f}",
            "% with entity overlap":   f"{has_entity.mean() * 100:.1f}",
        })
    _save_table(pd.DataFrame(rows), "table_dataset_stats", outdir)


# ---------------------------------------------------------------------------
# Console summaries
# ---------------------------------------------------------------------------

def _print_kw_summary(df: pd.DataFrame) -> None:
    print("\n=== Kruskal-Wallis significant results (p < 0.05) ===")
    for group_col in ("caption_type", "category", "source"):
        if group_col not in df.columns:
            continue
        groups_vals = df[group_col].dropna().unique()
        for sim in SIM_COLS:
            arrays = [df[df[group_col] == g][sim].dropna().values for g in groups_vals]
            H, p, eta2 = _kruskal_eta2(arrays)
            if not np.isnan(p) and p < 0.05:
                print(f"  {group_col} × {SIM_LABELS[sim]}: "
                      f"H={H:.2f}, p={p:.4f}, η²={eta2:.3f}")
            print(f"  [eta²] {group_col} × {SIM_LABELS[sim]}: η²={eta2:.3f}")


def _print_hypothesis_assessment(df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    print("\n=== Hypothesis assessment (sample) ===")

    # H1: any structural feature significantly correlates with similarity
    sig = [(row["feature"], sim, row[f"r_{sim}"])
           for _, row in corr_df.iterrows()
           for sim in SIM_COLS
           if not np.isnan(row[f"p_{sim}"]) and row[f"p_{sim}"] < 0.05]
    if sig:
        best = max(sig, key=lambda x: abs(x[2]))
        print(f"  H1: {len(sig)} significant structural correlations found. "
              f"Strongest: {best[0]} × {SIM_LABELS[best[1]]} "
              f"(r={best[2]:.2f}) — consistent with H1")
    else:
        print("  H1: No significant structural correlations in sample — insufficient evidence")

    # H2: caption_type affects similarity
    if "caption_type" in df.columns and (df["caption_type"] != "unclassified").any():
        groups = df["caption_type"].dropna().unique()
        arrays = [df[df["caption_type"] == g]["sbert_sim"].dropna().values for g in groups]
        H, p, _ = _kruskal_eta2(arrays)
        verdict = "consistent" if (not np.isnan(p) and p < 0.05) else "not yet supported"
        print(f"  H2: KW(sbert_sim by caption_type) H={H:.2f}, p={p:.3f} — {verdict}")
    else:
        print("  H2: caption_type not yet classified — cannot assess H2")

    # H3: entity type overlap comparison
    overlap_means = {etype: df[col].mean() for etype, col in zip(ENTITY_TYPES, OVERLAP_COLS)}
    ranked = sorted(overlap_means.items(), key=lambda x: x[1], reverse=True)
    top_type, top_val = ranked[0]
    bot_type, bot_val = ranked[-1]
    print(f"  H3: {top_type} overlap (mean={top_val:.3f}) > "
          f"{bot_type} overlap (mean={bot_val:.3f}) — "
          f"{'consistent' if top_val > bot_val else 'not consistent'} with H3")

    # H4: category affects similarity
    groups = df["category"].dropna().unique()
    arrays = [df[df["category"] == g]["sbert_sim"].dropna().values for g in groups]
    H, p, _ = _kruskal_eta2(arrays)
    verdict = "consistent" if (not np.isnan(p) and p < 0.05) else "not yet supported"
    print(f"  H4: KW(sbert_sim by category) H={H:.2f}, p={p:.3f} — {verdict}")

    # H5: UK (guardian, bbc) vs US (wapo, usatoday) entity Jaccard
    uk = df[df["source"].isin(["guardian", "bbc"])]["sbert_sim"].dropna()
    us = df[df["source"].isin(["wapo", "usatoday"])]["sbert_sim"].dropna()
    if len(uk) > 0 and len(us) > 0:
        verdict = "consistent" if uk.mean() > us.mean() else "not consistent"
        print(f"  H5: UK mean sbert_sim={uk.mean():.3f}, "
              f"US mean sbert_sim={us.mean():.3f} — {verdict} with H5")
    else:
        print("  H5: insufficient source data to assess")


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

REQUIRED_COLS = (
    ["item_id", "source", "category", "caption", "article_lead", "word_count"]
    + SIM_COLS
    + OVERLAP_COLS
    + ["entity_jaccard", "entity_coverage", "caption_type"]
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper figures and tables from pipeline results"
    )
    parser.add_argument("--input",  required=True, help="Pipeline output CSV")
    parser.add_argument("--output", required=True, help="Output directory for figures/tables")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    _setup_style()

    saved: list[str] = []

    # --- Descriptive ---
    fig_caption_length(df, args.output, saved)
    fig_similarity_distributions(df, args.output, saved)
    table_dataset_stats(df, args.output)

    # --- RQ1 ---
    corr_df = _compute_correlations(df)
    fig_rq1_heatmap(corr_df, args.output, saved)
    fig_rq1_top_scatter(df, corr_df, args.output, saved)
    table_rq1(corr_df, args.output)

    # --- RQ2 ---
    if "caption_type" in df.columns and (df["caption_type"] != "unclassified").any():
        fig_rq2(df, args.output, saved)
        table_rq2(df, args.output)
    else:
        print("\nWARNING: caption_type is all 'unclassified'; skipping RQ2 figures.")

    # --- RQ3 ---
    fig_rq3_entity_bar(df, args.output, saved)
    fig_rq3_entity_scatter(df, args.output, saved)
    table_rq3(df, args.output)

    # --- RQ4 ---
    fig_rq4(df, args.output, saved)
    table_rq4(df, args.output)

    # --- RQ5 ---
    fig_rq5_by_source(df, args.output, saved)
    fig_rq5_entity_bar(df, args.output, saved)
    table_rq5(df, args.output)
    table_rq5_dunn(df, args.output)

    # --- Console summary ---
    print("\n=== Saved figures ===")
    for path in saved:
        print(f"  {path}")

    _print_kw_summary(df)
    _print_hypothesis_assessment(df, corr_df)


if __name__ == "__main__":
    main()