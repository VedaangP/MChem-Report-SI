#!/usr/bin/env python3
"""
pareto_analysis.py
------------------
Pareto frontier analysis + cross-dataset consensus for feature-selection
CSV files produced by the halogen-bond ML pipeline.

Usage
-----
  python pareto_analysis.py Cl_all_combinations.csv Br_all_combinations.csv ...
  python pareto_analysis.py results/*_all_combinations.csv

Per-dataset outputs  (./pareto_output/)
---------------------------------------
  {tag}_composite_ranking.png
  {tag}_pareto_frontier.png
  {tag}_analysis.txt

Cross-dataset outputs
----------------------
  combined_pareto_frontier.png
  cross_dataset_universal_ranking.png  — top-20 feature sets by universal score
  cross_dataset_heatmap.png            — composite score per model × acceptor
  cross_dataset_valr2_heatmap.png      — Val R² per model × acceptor
  cross_dataset_feature_frequency.png  — individual descriptor consensus
  cross_dataset_analysis.txt           — full cross-dataset text report
  cross_dataset_rankings.csv           — machine-readable universal rankings
  cross_dataset_feature_freq.csv       — machine-readable feature frequencies
  holdout_evaluation.txt               — CLEAN test-set report (printed after selection)

Test_R2 and Test_RMSE removed from WEIGHTS; they are never used during
feature-set selection.  A new holdout_report() reads those columns back
for the single chosen feature set only, *after* the universal set is locked.

Three aggregation methods are ensembled for the Universal_Score:
    • RRF   (Reciprocal Rank Fusion, k=60)  — weight 0.40
    • Borda count                            — weight 0.40
    • MinMax (original coverage+mean+min+consistency) — weight 0.20"""

import sys
import glob
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("pareto_output")

# Selection weights use ONLY train/val metrics — no test data.
# Test_R2 / Test_RMSE remain in the CSV and are displayed in reports, but
# they play no role in ranking or composite computation.
WEIGHTS = {
    "Val_R2_CV":        0.35,   # primary generalisation signal (CV)
    "Gap":              0.25,   # train-val gap — most direct overfitting guard
    "BIC":              0.18,   # complexity penalty
    "Adj_R2":           0.10,   # in-sample fit quality (adjusted)
    "CV_RMSE":          0.09,   # error magnitude (CV)
    "N_desc_unstable":  0.03,   # stability penalty
}

# Weights for the MinMax component of the universal score
UNIVERSAL_WEIGHTS = {
    "coverage":       0.25,   # fraction of datasets this feature set appears in
    "mean_composite": 0.40,   # mean composite across datasets
    "min_composite":  0.20,   # worst-case performance (robustness)
    "consistency":    0.15,   # 1 - normalised std  (lower variance = better)
}

# Ensemble aggregation weights
RRF_K = 60          # Reciprocal Rank Fusion smoothing constant (standard value)
AGGR_WEIGHTS = {
    "rrf":    0.40,
    "borda":  0.40,
    "minmax": 0.20,
}

TOP_N_FREQ = 20   # top-N models per dataset used for feature-frequency analysis

COLORWAY = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#B07AA1", "#9D755D", "#BAB0AC", "#72B7B2",
    "#FF9DA7", "#EDC948",
    ]

FEAT_COLORS = {
    1: "#aec7e8", 2: "#ffbb78", 3: "#98df8a", 4: "#F58518",
    5: "#4C78A8", 6: "#E45756", 7: "#B07AA1", 8: "#9D755D", 9: "#72B7B2",
}

DISPLAY_COLS = [
    "N_features", "Features", "Val_R2_CV", "Test_R2",
    "Gap", "Test_RMSE", "BIC", "Adj_R2", "N_desc_unstable", "Composite",
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def normalize_features(feat_str: str) -> str:
    """
    Sort individual feature names alphabetically so that 'A, B' and 'B, A'
    map to the same canonical key, enabling reliable cross-dataset matching.
    """
    if not isinstance(feat_str, str):
        return feat_str
    return ", ".join(sorted(f.strip() for f in feat_str.split(",")))


def count_desc_unstable(s: str) -> int:
    if not isinstance(s, str) or s.strip() in ("—", ""):
        return 0
    return sum(1 for x in s.split(",") if "Scaffold" not in x)


def _norm_high(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)


def _norm_low(s: pd.Series) -> pd.Series:
    return 1.0 - _norm_high(s)


def compute_composite(df: pd.DataFrame) -> pd.Series:
    """
    Selection composite uses ONLY train/val metrics.
    Test_R2 and Test_RMSE are intentionally excluded.
    """
    s = pd.DataFrame(index=df.index)
    s["Val_R2_CV"]       = _norm_high(df["Val_R2_CV"])
    s["Gap"]             = _norm_low(df["Gap"])
    s["BIC"]             = _norm_low(df["BIC"])
    s["Adj_R2"]          = _norm_high(df["Adj_R2"])
    s["CV_RMSE"]         = _norm_low(df["CV_RMSE"])
    s["N_desc_unstable"] = _norm_low(df["N_desc_unstable"])
    return sum(s[m] * w for m, w in WEIGHTS.items())


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"N_features", "Features", "Train_R2_CV", "Val_R2_CV",
                "Gap", "CV_RMSE", "Adj_R2", "BIC", "Test_R2", "Test_RMSE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if "Unstable" not in df.columns:
        df["Unstable"] = "—"
    df["N_desc_unstable"] = df["Unstable"].apply(count_desc_unstable)
    df["Features"]        = df["Features"].apply(normalize_features)
    df["Composite"]       = compute_composite(df)
    df["Composite_Rank"]  = df["Composite"].rank(ascending=False).astype(int)
    return df


def best_per_n(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values("Composite", ascending=False)
          .groupby("N_features", sort=True)
          .first()
          .reset_index()
    )


# ── Text reports ──────────────────────────────────────────────────────────────

def text_report(df: pd.DataFrame, tag: str) -> str:
    bpn   = best_per_n(df)
    top15 = df.sort_values("Composite", ascending=False).head(15)
    win   = bpn.loc[bpn["Composite"].idxmax()]

    lines = [
        "=" * 100,
        f"  PARETO ANALYSIS — {tag}   ({len(df)} models)",
        f"  NOTE: Composite score uses Val_R2_CV, Gap, BIC, Adj_R2, CV_RMSE only.",
        f"        Test_R2 / Test_RMSE are shown for reference but NOT used in selection.",
        "=" * 100,
        "",
        "── BEST MODEL PER FEATURE COUNT ─────────────────────────────────────────────",
        bpn[DISPLAY_COLS].to_string(index=False),
        "",
        "── TOP 15 BY COMPOSITE SCORE ────────────────────────────────────────────────",
        top15[DISPLAY_COLS].to_string(index=False),
        "",
        "── PARSIMONIOUS OPTIMUM ─────────────────────────────────────────────────────",
        f"  Features   : {win['Features']}",
        f"  N_features : {int(win['N_features'])}",
        f"  Val R²     : {win['Val_R2_CV']:.4f}",
        f"  Test R²    : {win['Test_R2']:.4f}  (display only — not used in selection)",
        f"  Gap        : {win['Gap']:.4f}",
        f"  Test RMSE  : {win['Test_RMSE']:.4f} kcal/mol  (display only)",
        f"  BIC        : {win['BIC']:.2f}",
        f"  Composite  : {win['Composite']:.4f}",
        "",
        f"  Composite weights: {WEIGHTS}",
        "",
    ]
    return "\n".join(lines)


def text_cross_report(cross_df: pd.DataFrame, full_cov: pd.DataFrame,
                      freq_df: pd.DataFrame, tags: list) -> str:
    n = len(tags)
    cross_cols = [
        "N_features", "Features", "Coverage",
        "Mean_Composite", "Min_Composite", "Std_Composite",
        "Mean_Val_R2", "Min_Val_R2", "Mean_Test_R2",
        "Score_RRF", "Score_Borda", "Score_MinMax",
        "Rank_RRF", "Rank_Borda", "Rank_MinMax", "Rank_Spread",
        "Universal_Score",
    ]
    # only keep cols that exist
    cross_cols = [c for c in cross_cols if c in cross_df.columns]

    lines = [
        "=" * 100,
        f"  CROSS-DATASET ANALYSIS — {n} datasets: {', '.join(tags)}",
        "=" * 100,
        "",
        "── AGGREGATION METHOD ────────────────────────────────────────────────────────",
        f"  Universal_Score = RRF(0.40) + Borda(0.40) + MinMax(0.20)",
        f"  RRF   : sum_i [ 1 / ({RRF_K} + rank_i) ]  — absent datasets contribute 0",
        f"  Borda : sum_i [ N_models - rank_i ]        — absent datasets contribute 0",
        f"  MinMax: coverage(0.25) + mean_composite(0.40) + min_composite(0.20) + consistency(0.15)",
        f"  Rank_Spread = max(Rank_RRF, Rank_Borda, Rank_MinMax) - min(...)  [low = robust]",
        "",
        "── TOP 20 BY UNIVERSAL SCORE ─────────────────────────────────────────────────",
        cross_df.head(20)[cross_cols].to_string(index=False),
        "",
    ]

    if len(full_cov) > 0:
        lines += [
            f"── FULL COVERAGE (present in all {n} datasets) — TOP 10 ──────────────────────",
            full_cov.head(10)[cross_cols].to_string(index=False),
            "",
            "── ★  RECOMMENDED UNIVERSAL PARAMETER SET ───────────────────────────────────",
        ]
        best = full_cov.iloc[0]
        lines += [
            f"  Features        : {best['Features']}",
            f"  N_features      : {int(best['N_features'])}",
            f"  Coverage        : {int(best['Coverage'])}/{n} datasets",
            f"  Mean Composite  : {best['Mean_Composite']:.4f}",
            f"  Min Composite   : {best['Min_Composite']:.4f}  (worst-case acceptor)",
            f"  Std Composite   : {best['Std_Composite']:.4f}",
            f"  Mean Val R²     : {best['Mean_Val_R2']:.4f}",
            f"  Min Val R²      : {best['Min_Val_R2']:.4f}  (worst-case acceptor)",
            f"  Score RRF       : {best['Score_RRF']:.4f}",
            f"  Score Borda     : {best['Score_Borda']:.4f}",
            f"  Score MinMax    : {best['Score_MinMax']:.4f}",
            f"  Rank Spread     : {int(best['Rank_Spread'])}  (0 = all methods agree perfectly)",
            f"  Universal Score : {best['Universal_Score']:.4f}",
            "",
            "  Per-dataset breakdown (Val/Composite — test shown for reference only):",
        ]
        for tag in tags:
            c = best.get(f"Composite_{tag}", float("nan"))
            v = best.get(f"Val_R2_{tag}",   float("nan"))
            t = best.get(f"Test_R2_{tag}",  float("nan"))
            lines.append(
                f"    {tag:12s}  Composite={c:.4f}  Val R²={v:.4f}"
                f"  Test R²={t:.4f} (ref only)"
            )
    else:
        lines += [
            "  ⚠  No single feature set appears in ALL datasets.",
            "     Use the top of the universal ranking above as a near-universal choice,",
            "     or expand the feature-combination search to include missing acceptors.",
        ]

    lines += [
        "",
        f"── TOP 30 INDIVIDUAL DESCRIPTORS BY CONSENSUS FREQUENCY ─────────────────────",
        f"   (from top-{TOP_N_FREQ} models per dataset × {n} datasets; max count = {n * TOP_N_FREQ})",
        freq_df.head(30).to_string(index=False),
        "",
    ]
    return "\n".join(lines)


# ── Holdout report ────────────────────────────────────────────────────────────

def holdout_report(selected_features: str, all_dfs: dict, tags: list) -> str:
    """
    Called ONCE after the universal feature set is locked.

    Reads Test_R2 / Test_RMSE from the pre-computed CSVs for the selected
    feature set only.  These numbers were NEVER used during selection, so
    this constitutes a clean, uncontaminated final evaluation.

    Parameters
    ----------
    selected_features : canonical (sorted) feature string, e.g.
                        "C6, Dipole_1, Monopole, Polarizability, Q_2, r2"
    all_dfs           : dict {tag: DataFrame} loaded by load_csv()
    tags              : ordered list of dataset tags
    """
    lines = [
        "=" * 100,
        "  HOLDOUT EVALUATION — selected universal feature set",
        "  Test_R2 / Test_RMSE were never used during feature selection.",
        "  This report is produced only AFTER the universal set is locked.",
        "=" * 100,
        f"  Feature set : {selected_features}",
        "",
        f"  {'Dataset':<15} {'Val R² (CV)':>12} {'Test R²':>10} "
        f"{'Test RMSE':>12} {'Gap':>8} {'BIC':>10}",
        "  " + "-" * 72,
    ]
    test_r2s, test_rmses = [], []
    for tag in tags:
        df  = all_dfs[tag]
        row = df[df["Features"] == selected_features]
        if row.empty:
            lines.append(
                f"  {tag:<15} {'—':>12} {'—':>10} {'—':>12} {'—':>8} {'—':>10}"
                f"  ← feature set absent from this dataset"
            )
            continue
        r = row.iloc[0]
        lines.append(
            f"  {tag:<15} {r['Val_R2_CV']:>12.4f} {r['Test_R2']:>10.4f}"
            f" {r['Test_RMSE']:>12.4f} {r['Gap']:>8.4f} {r['BIC']:>10.2f}"
        )
        test_r2s.append(r["Test_R2"])
        test_rmses.append(r["Test_RMSE"])

    if test_r2s:
        lines += [
            "  " + "-" * 72,
            f"  {'Mean':<15} {'':>12} {np.mean(test_r2s):>10.4f}"
            f" {np.mean(test_rmses):>12.4f}",
            f"  {'Std':<15} {'':>12} {np.std(test_r2s):>10.4f}"
            f" {np.std(test_rmses):>12.4f}",
            f"  {'Min (worst-case)':<15} {'':>12} {np.min(test_r2s):>10.4f}"
            f" {np.max(test_rmses):>12.4f}",
        ]
    lines.append("")
    return "\n".join(lines)


# ── Cross-dataset aggregation ─────────────────────────────────────────────────

def _reciprocal_rank_fusion(cross_df: pd.DataFrame,
                             all_dfs: dict, k: int = RRF_K) -> pd.Series:
    """
    RRF: for each (feature_set, dataset) pair find the composite
    rank within that dataset and accumulate 1/(k + rank).
    Feature sets absent from a dataset contribute 0, naturally penalising
    low-coverage sets without needing a separate coverage term.
    """
    rrf_scores  = pd.Series(0.0, index=cross_df.index)
    feat_index  = cross_df["Features"].values

    for tag, df in all_dfs.items():
        ranks = df.set_index("Features")["Composite_Rank"].to_dict()
        for i, feat in enumerate(feat_index):
            r = ranks.get(feat, None)
            if r is not None:
                rrf_scores.iloc[i] += 1.0 / (k + r)
    return rrf_scores


def _borda_count(cross_df: pd.DataFrame, all_dfs: dict) -> pd.Series:
    """
    Borda count: award (N_models - rank_i) points per dataset.
    Absent sets score 0, rewarding broad coverage automatically.
    """
    borda_scores = pd.Series(0.0, index=cross_df.index)
    feat_index   = cross_df["Features"].values

    for tag, df in all_dfs.items():
        n_models = len(df)
        ranks    = df.set_index("Features")["Composite_Rank"].to_dict()
        for i, feat in enumerate(feat_index):
            r = ranks.get(feat, None)
            if r is not None:
                borda_scores.iloc[i] += (n_models - r)
    return borda_scores


def cross_dataset_analysis(all_dfs: dict):
    """
    Aggregate per-feature-set performance across all loaded datasets and
    rank by a universal score that ensembles RRF, Borda count, and the
    original MinMax (coverage + mean + min + consistency) approach.

    Returns
    -------
    cross_df  : DataFrame — all feature sets ranked by Universal_Score
    full_cov  : DataFrame — subset that appears in every dataset
    freq_df   : DataFrame — individual descriptor frequency in top-N models
    """
    tags = list(all_dfs.keys())
    n_ds = len(tags)

    # ── aggregate per feature set ─────────────────────────────────────────────
    records: dict = {}
    for tag, df in all_dfs.items():
        for _, row in df.iterrows():
            feat = row["Features"]
            if feat not in records:
                records[feat] = {"N_features": int(row["N_features"]), "scores": {}}
            records[feat]["scores"][tag] = {
                "Composite": row["Composite"],
                "Val_R2_CV": row["Val_R2_CV"],
                "Test_R2":   row["Test_R2"],
                "Gap":       row["Gap"],
                "Test_RMSE": row["Test_RMSE"],
                "BIC":       row["BIC"],
            }

    rows = []
    for feat, info in records.items():
        sc         = info["scores"]
        coverage   = len(sc)
        composites = [v["Composite"] for v in sc.values()]
        val_r2s    = [v["Val_R2_CV"] for v in sc.values()]
        test_r2s   = [v["Test_R2"]   for v in sc.values()]

        r = {
            "Features":       feat,
            "N_features":     info["N_features"],
            "Coverage":       coverage,
            "Mean_Composite": np.mean(composites),
            "Min_Composite":  np.min(composites),
            "Std_Composite":  np.std(composites),
            "Mean_Val_R2":    np.mean(val_r2s),
            "Min_Val_R2":     np.min(val_r2s),
            "Mean_Test_R2":   np.mean(test_r2s),   # kept for display / holdout
            "Min_Test_R2":    np.min(test_r2s),
        }
        for tag in tags:
            r[f"Composite_{tag}"] = sc.get(tag, {}).get("Composite", np.nan)
            r[f"Val_R2_{tag}"]    = sc.get(tag, {}).get("Val_R2_CV", np.nan)
            r[f"Test_R2_{tag}"]   = sc.get(tag, {}).get("Test_R2",   np.nan)
        rows.append(r)

    cross_df = pd.DataFrame(rows)

    # ── MinMax component (original universal score) ───────────────────
    cross_df["_cov"]  = cross_df["Coverage"] / n_ds
    cross_df["_mean"] = _norm_high(cross_df["Mean_Composite"])
    cross_df["_min"]  = _norm_high(cross_df["Min_Composite"])
    cross_df["_cons"] = _norm_low(cross_df["Std_Composite"])
    cross_df["Score_MinMax"] = (
        cross_df["_cov"]  * UNIVERSAL_WEIGHTS["coverage"]       +
        cross_df["_mean"] * UNIVERSAL_WEIGHTS["mean_composite"]  +
        cross_df["_min"]  * UNIVERSAL_WEIGHTS["min_composite"]   +
        cross_df["_cons"] * UNIVERSAL_WEIGHTS["consistency"]
    )
    cross_df.drop(columns=["_cov", "_mean", "_min", "_cons"], inplace=True)

    # ── RRF component ─────────────────────────────────────────────────
    cross_df["Score_RRF"]   = _norm_high(_reciprocal_rank_fusion(cross_df, all_dfs))

    # ──Borda component ───────────────────────────────────────────────
    cross_df["Score_Borda"] = _norm_high(_borda_count(cross_df, all_dfs))

    # ── Ensemble universal score ─────────────────────────────────────
    cross_df["Universal_Score"] = (
        cross_df["Score_RRF"]    * AGGR_WEIGHTS["rrf"]    +
        cross_df["Score_Borda"]  * AGGR_WEIGHTS["borda"]  +
        cross_df["Score_MinMax"] * AGGR_WEIGHTS["minmax"]
    )
    cross_df.sort_values("Universal_Score", ascending=False, inplace=True)
    cross_df.reset_index(drop=True, inplace=True)

    # ──  Per-method ranks and spread (robustness indicator) ────────────
    for col, rank_col in [
        ("Score_RRF",    "Rank_RRF"),
        ("Score_Borda",  "Rank_Borda"),
        ("Score_MinMax", "Rank_MinMax"),
    ]:
        cross_df[rank_col] = cross_df[col].rank(ascending=False).astype(int)

    cross_df["Rank_Spread"] = (
        cross_df[["Rank_RRF", "Rank_Borda", "Rank_MinMax"]].max(axis=1) -
        cross_df[["Rank_RRF", "Rank_Borda", "Rank_MinMax"]].min(axis=1)
    )

    full_cov = cross_df[cross_df["Coverage"] == n_ds].copy()

    # ── individual descriptor frequency ──────────────────────────────────────
    feat_counts: dict = {}
    for tag, df in all_dfs.items():
        for feat_str in df.nlargest(TOP_N_FREQ, "Composite")["Features"]:
            for f in feat_str.split(","):
                f = f.strip()
                if f:
                    feat_counts[f] = feat_counts.get(f, 0) + 1

    freq_df = pd.DataFrame(
        sorted(feat_counts.items(), key=lambda x: -x[1]),
        columns=["Feature", "Count"],
    )
    freq_df["Pct_of_TopN"] = (
        freq_df["Count"] / (n_ds * TOP_N_FREQ) * 100
    ).round(1)

    return cross_df, full_cov, freq_df


# ── Cross-dataset plotting ────────────────────────────────────────────────────

def save_meta(path: Path, caption: str, description: str = "") -> None:
    with open(str(path) + ".meta.json", "w") as f:
        json.dump({"caption": caption, "description": description}, f)


def plot_universal_ranking(cross_df: pd.DataFrame, full_cov: pd.DataFrame,
                           tags: list) -> None:
    top20 = cross_df.head(20).copy().iloc[::-1]
    n_ds  = len(tags)

    colors = []
    for _, row in top20.iterrows():
        if row["Coverage"] == n_ds:
            colors.append("#54A24B")
        elif row["Coverage"] >= n_ds - 1:
            colors.append("#F58518")
        else:
            colors.append("#4C78A8")

    labels = top20.apply(
        lambda r: (
            f"[{int(r['Coverage'])}/{n_ds}]  {r['Features']}  ({int(r['N_features'])}f)"
            f"  spread={int(r['Rank_Spread'])}"
        ),
        axis=1,
    ).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=list(range(len(top20))),
        x=top20["Universal_Score"],
        orientation="h",
        marker_color=colors,
        text=[f"{s:.3f}" for s in top20["Universal_Score"]],
        textposition="outside",
        customdata=top20[[
            "Coverage", "Mean_Composite", "Min_Composite",
            "Std_Composite", "Mean_Val_R2", "Min_Val_R2",
            "Score_RRF", "Score_Borda", "Score_MinMax", "Rank_Spread",
        ]].values,
        hovertemplate=(
            "Coverage: %{customdata[0]}<br>"
            "Mean Composite: %{customdata[1]:.4f}<br>"
            "Min Composite:  %{customdata[2]:.4f}<br>"
            "Std Composite:  %{customdata[3]:.4f}<br>"
            "Mean Val R²:    %{customdata[4]:.4f}<br>"
            "Min Val R²:     %{customdata[5]:.4f}<br>"
            "Score RRF:      %{customdata[6]:.4f}<br>"
            "Score Borda:    %{customdata[7]:.4f}<br>"
            "Score MinMax:   %{customdata[8]:.4f}<br>"
            "Rank Spread:    %{customdata[9]}<extra></extra>"
        ),
    ))
    for col, lbl in [
        ("#54A24B", f"Full coverage ({n_ds}/{n_ds})"),
        ("#F58518", f"Near-full ({n_ds - 1}/{n_ds})"),
        ("#4C78A8", "Partial coverage"),
    ]:
        fig.add_trace(go.Bar(y=[None], x=[None], orientation="h",
                             marker_color=col, name=lbl))

    fig.update_yaxes(
        tickvals=list(range(len(top20))),
        ticktext=labels,
        title_text="Feature set  [coverage | spread = rank agreement across methods]",
    )
    fig.update_xaxes(title_text="Universal score", range=[0, 1.25])
    fig.update_layout(
        title={
            "text": (
                "Cross-dataset universal ranking — top 20 feature sets<br>"
                f"<span style='font-size:14px;font-weight:normal'>"
                f"RRF(0.40) + Borda(0.40) + MinMax(0.20) | "
                f"datasets: {', '.join(tags)}</span>"
            )
        },
        legend=dict(orientation="h", yanchor="bottom", y=1.07,
                    xanchor="center", x=0.5),
        barmode="overlay",
        height=max(600, 44 * len(top20) + 220),
        margin=dict(l=540),
    )
    fig.update_traces(cliponaxis=False)
    out = OUT_DIR / "cross_dataset_universal_ranking.png"
    fig.write_image(str(out))
    save_meta(
        out,
        "Cross-dataset universal ranking — top 20 feature sets",
        "Green = full coverage; Orange = missing one dataset; Blue = partial. "
        "Spread shows agreement across RRF / Borda / MinMax methods.",
    )
    print(f"  Saved: {out}")


def plot_cross_heatmap(cross_df: pd.DataFrame, tags: list,
                       n_top: int = 25) -> None:
    top = cross_df.head(n_top).copy().iloc[::-1]

    def _short(feat, n):
        s = f"[{n}] {feat}"
        return (s[:60] + "…") if len(s) > 60 else s

    y_labels = [
        _short(r["Features"], int(r["N_features"]))
        for _, r in top.iterrows()
    ]

    for metric, colorscale, cbar_title, out_name in [
        ("Composite", "RdYlGn", "Composite score",  "cross_dataset_heatmap.png"),
        ("Val_R2",    "Blues",  "Val R² (CV)",       "cross_dataset_valr2_heatmap.png"),
    ]:
        z = [
            [row.get(f"{metric}_{t}", np.nan) for t in tags]
            for _, row in top.iterrows()
        ]
        text = [
            [f"{v:.3f}" if not np.isnan(v) else "—" for v in row]
            for row in z
        ]

        fig = go.Figure(go.Heatmap(
            z=z, x=tags, y=y_labels,
            colorscale=colorscale,
            zmin=0 if metric == "Composite" else None,
            zmax=1 if metric == "Composite" else None,
            text=text, texttemplate="%{text}",
            hovertemplate=(
                "Acceptor: %{x}<br>Feature set: %{y}<br>"
                + cbar_title + ": %{z:.4f}<extra></extra>"
            ),
            colorbar=dict(title=cbar_title),
        ))
        fig.update_layout(
            title={
                "text": (
                    f"Cross-dataset {cbar_title} — top {n_top} universal feature sets<br>"
                    "<span style='font-size:13px;font-weight:normal'>"
                    "Grey/NaN cells = feature set not evaluated for that acceptor</span>"
                )
            },
            xaxis_title="Acceptor",
            yaxis_title="Feature set",
            height=max(500, 28 * n_top + 200),
            margin=dict(l=500),
        )
        out = OUT_DIR / out_name
        fig.write_image(str(out))
        save_meta(out, f"Cross-dataset {cbar_title} heatmap — top {n_top} universal feature sets")
        print(f"  Saved: {out}")


def plot_feature_frequency(freq_df: pd.DataFrame, tags: list,
                           n_top: int = 30) -> None:
    top = freq_df.head(n_top).sort_values("Count")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top["Feature"],
        x=top["Count"],
        orientation="h",
        marker=dict(
            color=top["Count"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Count"),
        ),
        text=[f"{c}  ({p:.0f}%)" for c, p in zip(top["Count"], top["Pct_of_TopN"])],
        textposition="outside",
        hovertemplate="Descriptor: %{y}<br>Count: %{x}<extra></extra>",
    ))
    max_count = int(top["Count"].max())
    fig.update_xaxes(
        title_text=(
            f"Occurrences across top-{TOP_N_FREQ} models "
            f"× {len(tags)} datasets  (max = {len(tags) * TOP_N_FREQ})"
        ),
        range=[0, max_count * 1.28],
    )
    fig.update_yaxes(title_text="Descriptor")
    fig.update_layout(
        title={
            "text": (
                f"Consensus descriptor frequency — top {n_top}<br>"
                f"<span style='font-size:14px;font-weight:normal'>"
                f"datasets: {', '.join(tags)}</span>"
            )
        },
        height=max(500, 26 * n_top + 200),
        margin=dict(l=260),
    )
    fig.update_traces(cliponaxis=False)
    out = OUT_DIR / "cross_dataset_feature_frequency.png"
    fig.write_image(str(out))
    save_meta(out, f"Consensus descriptor frequency across {len(tags)} datasets")
    print(f"  Saved: {out}")


# ── Per-dataset plotting ──────────────────────────────────────────────────────

def plot_composite_ranking(df: pd.DataFrame, tag: str) -> None:
    top15 = df.sort_values("Composite", ascending=False).head(15).copy()
    top15["Label"] = top15.apply(
        lambda r: f"#{int(r['Composite_Rank'])}  {r['Features']}  ({int(r['N_features'])}f)",
        axis=1,
    )
    top15 = top15.iloc[::-1]
    colors = [FEAT_COLORS.get(n, "#BAB0AC") for n in top15["N_features"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=list(range(len(top15))),
        x=top15["Composite"],
        orientation="h",
        marker_color=colors,
        text=[f"{c:.3f}" for c in top15["Composite"]],
        textposition="outside",
        customdata=top15[["Features", "Val_R2_CV", "Test_R2", "Gap", "Test_RMSE"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Val R²: %{customdata[1]:.4f} | Test R² (ref): %{customdata[2]:.4f}<br>"
            "Gap: %{customdata[3]:.4f} | Test RMSE (ref): %{customdata[4]:.4f}"
            "<extra></extra>"
        ),
    ))
    for n, c in sorted(FEAT_COLORS.items()):
        fig.add_trace(go.Bar(y=[None], x=[None], orientation="h",
                             marker_color=c, name=f"{n} features"))
    fig.update_yaxes(
        tickvals=list(range(len(top15))),
        ticktext=top15["Label"].tolist(),
        title_text="Model",
    )
    fig.update_xaxes(title_text="Composite score (val/train only)", range=[0, 1.12])
    fig.update_layout(
        title={"text": f"Top 15 models — {tag}<br>"
                       "<span style='font-size:16px;font-weight:normal'>"
                       "Weights: Val R²(0.35) Gap(0.25) BIC(0.18) Adj_R²(0.10) CV_RMSE(0.09)…"
                       "</span>"},
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                    xanchor="center", x=0.5),
        barmode="overlay",
    )
    fig.update_traces(cliponaxis=False)
    out = OUT_DIR / f"{tag}_composite_ranking.png"
    fig.write_image(str(out))
    save_meta(out, f"Top-15 models by composite score — {tag}")
    print(f"  Saved: {out}")


def plot_pareto_frontier(df: pd.DataFrame, tag: str) -> dict:
    bpn = best_per_n(df)
    nf  = bpn["N_features"].tolist()
    val = bpn["Val_R2_CV"].tolist()
    tst = bpn["Test_R2"].tolist()
    bic = bpn["BIC"].tolist()

    win     = bpn.loc[bpn["Composite"].idxmax()]
    win_n   = int(win["N_features"])
    win_idx = nf.index(win_n)

    def msizes(wi, total, large=15, small=8):
        return [large if i == wi else small for i in range(total)]

    def mcolors(wi, total, wc, bc):
        return [wc if i == wi else bc for i in range(total)]

    bic_arr  = np.array(bic, dtype=float)
    bic_norm = (bic_arr - bic_arr.min()) / max(bic_arr.max() - bic_arr.min(), 1e-9)
    bic_sc   = bic_norm * 0.25 + 0.68

    fig = go.Figure()
    fig.add_trace(go.Bar(x=nf, y=bic_sc, name="BIC (scaled)",
                         marker_color="rgba(160,160,160,0.22)"))
    fig.add_trace(go.Scatter(
        x=nf, y=val, name="Val R² (CV)",
        mode="lines+markers",
        line=dict(width=2.5, color="steelblue"),
        marker=dict(size=msizes(win_idx, len(nf)),
                    color=mcolors(win_idx, len(nf), "green", "steelblue"),
                    line=dict(color="white", width=1)),
        customdata=bpn[["Features", "Gap", "Test_RMSE"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Val R²: %{y:.4f} | Gap: %{customdata[1]:.4f}<br>"
            "Test RMSE (ref): %{customdata[2]:.4f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=nf, y=tst, name="Test R² (ref only)",
        mode="lines+markers",
        line=dict(width=2.5, dash="dash", color="tomato"),
        marker=dict(size=msizes(win_idx, len(nf)),
                    color=mcolors(win_idx, len(nf), "green", "tomato"),
                    line=dict(color="white", width=1)),
        customdata=bpn[["Features", "Gap", "Test_RMSE"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Test R² (ref): %{y:.4f} | Gap: %{customdata[1]:.4f}<br>"
            "Test RMSE (ref): %{customdata[2]:.4f}<extra></extra>"
        ),
    ))

    wrapped = "<br>".join(textwrap.wrap(win["Features"], 28))
    fig.add_annotation(
        x=win_n, y=float(win["Val_R2_CV"]),
        text=f"★ Optimum ({win_n}f)<br>{wrapped}",
        showarrow=True, arrowhead=2, ax=60, ay=45,
        font=dict(size=10, color="green"),
        bgcolor="white", bordercolor="green", borderwidth=1,
    )
    fig.update_xaxes(title_text="Number of features", tickvals=nf)
    fig.update_yaxes(title_text="R² score", range=[0.65, 1.04])
    fig.update_layout(
        title={"text": f"Pareto frontier — {tag}<br>"
                       "<span style='font-size:16px;font-weight:normal'>"
                       "Green = parsimonious optimum | grey bars = relative BIC | "
                       "dashed = Test R² (reference, not used in selection)</span>"},
        legend=dict(orientation="h", yanchor="bottom", y=1.07,
                    xanchor="center", x=0.5),
        barmode="overlay",
    )
    out = OUT_DIR / f"{tag}_pareto_frontier.png"
    fig.write_image(str(out))
    save_meta(out, f"Pareto frontier — {tag}")
    print(f"  Saved: {out}")
    return {"tag": tag, "nf": nf, "val": val, "tst": tst, "win_n": win_n}


def plot_combined_frontier(all_data: list) -> None:
    if len(all_data) < 2:
        return
    fig = go.Figure()
    for i, d in enumerate(all_data):
        color = COLORWAY[i % len(COLORWAY)]
        win_n = d["win_n"]
        fig.add_trace(go.Scatter(
            x=d["nf"], y=d["val"],
            name=d["tag"],
            mode="lines+markers",
            line=dict(width=2.5, color=color),
            marker=dict(
                size=[14 if n == win_n else 8 for n in d["nf"]],
                symbol=["star" if n == win_n else "circle" for n in d["nf"]],
                color=color,
                line=dict(color="white", width=1),
            ),
        ))
    all_nf = sorted({n for d in all_data for n in d["nf"]})
    fig.update_xaxes(title_text="Number of features", tickvals=all_nf)
    fig.update_yaxes(title_text="Val R² (CV)", range=[0.65, 1.01])
    fig.update_layout(
        title={"text": "Combined pareto frontier — all datasets<br>"
                       "<span style='font-size:16px;font-weight:normal'>"
                       "Stars = parsimonious optimum per dataset (val/train composite)</span>"},
        legend=dict(orientation="h", yanchor="bottom", y=1.07,
                    xanchor="center", x=0.5),
    )
    out = OUT_DIR / "combined_pareto_frontier.png"
    fig.write_image(str(out))
    save_meta(out, "Combined pareto frontier across all datasets")
    print(f"  Saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(file_paths: list) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    expanded = []
    for p in file_paths:
        matches = glob.glob(p)
        expanded.extend(matches if matches else [p])
    expanded = sorted(set(expanded))

    if not expanded:
        print("No files found. Pass CSV file paths as arguments.")
        sys.exit(1)

    all_frontier_data = []
    all_dfs: dict = {}

    for path in expanded:
        tag = Path(path).stem
        print(f"\n{'─' * 60}")
        print(f"Processing: {path}  (tag: {tag})")

        try:
            df = load_csv(path)
        except Exception as e:
            print(f"  ERROR loading {path}: {e}")
            continue

        print(f"  {len(df)} models loaded.")
        report = text_report(df, tag)
        print(report)
        (OUT_DIR / f"{tag}_analysis.txt").write_text(report)

        plot_composite_ranking(df, tag)
        fd = plot_pareto_frontier(df, tag)
        all_frontier_data.append(fd)
        all_dfs[tag] = df

    plot_combined_frontier(all_frontier_data)

    # ── Cross-dataset analysis ────────────────────────────────────────────────
    if len(all_dfs) >= 2:
        print(f"\n{'═' * 60}")
        print(f"Running cross-dataset analysis across: {', '.join(all_dfs)}")
        tags = list(all_dfs.keys())

        cross_df, full_cov, freq_df = cross_dataset_analysis(all_dfs)

        report = text_cross_report(cross_df, full_cov, freq_df, tags)
        print(report)
        (OUT_DIR / "cross_dataset_analysis.txt").write_text(report)

        cross_df.to_csv(OUT_DIR / "cross_dataset_rankings.csv",    index=False)
        freq_df.to_csv( OUT_DIR / "cross_dataset_feature_freq.csv", index=False)
        print(f"  Saved: {OUT_DIR}/cross_dataset_rankings.csv")
        print(f"  Saved: {OUT_DIR}/cross_dataset_feature_freq.csv")

        plot_universal_ranking(cross_df, full_cov, tags)
        plot_cross_heatmap(cross_df, tags)
        plot_feature_frequency(freq_df, tags)

        # ──  Holdout evaluation — runs AFTER selection is locked ───────
        print(f"\n{'═' * 60}")
        print("Generating holdout evaluation report …")

        if len(full_cov) > 0:
            chosen = full_cov.iloc[0]["Features"]
            print(f"  Universal feature set: {chosen}")
        else:
            chosen = cross_df.iloc[0]["Features"]
            print(f"  ⚠  No full-coverage set found. Using top universal set:")
            print(f"     {chosen}")

        h_report = holdout_report(chosen, all_dfs, tags)
        print(h_report)
        (OUT_DIR / "holdout_evaluation.txt").write_text(h_report)
        print(f"  Saved: {OUT_DIR}/holdout_evaluation.txt")

    print(f"\n{'=' * 60}")
    print(f"All outputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    main(sys.argv[1:])
