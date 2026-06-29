"""
Exploratory analysis: descriptive temporal analysis.

Compares rating and sentiment across temporal windows (pre/during/post)
for Cagliari (target) and Olbia (control), and contrasts 2026 event
windows against the 2025 baseline for Cagliari.

Outputs
-------
results/exploratory/temporal_rating_boxplot.png
results/exploratory/temporal_sentiment_boxplot.png
results/exploratory/temporal_cagliari_2026_vs_2025.png
results/exploratory/temporal_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.config import PROCESSED_DATA_DIR, RESULTS_DIR
from analysis.plot_style import PALETTE, WINDOW_LABELS as STYLE_WINDOW_LABELS, apply_style

apply_style()

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR        = os.path.join(RESULTS_DIR, "exploratory")
SENTIMENT_FILE = os.path.join(PROCESSED_DATA_DIR, "reviews_sentiment.csv")
TOPICS_FILE    = os.path.join(PROCESSED_DATA_DIR, "reviews_topics.csv")

EVENT_WINDOWS  = {"pre", "during", "post"}
TARGET_CITY    = "cagliari"
CONTROL_CITY   = "olbia"

WINDOW_ORDER   = ["pre", "during", "post"]
BASELINE_ORDER = ["baseline_pre", "baseline_during", "baseline_post"]
WINDOW_LABELS  = STYLE_WINDOW_LABELS
PALETTE_WINDOW = {"pre": PALETTE["pre"], "during": PALETTE["during"], "post": PALETTE["post"]}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(sample: int | None = None) -> pd.DataFrame:
    log.info("Loading sentiment + topics …")
    sent  = pd.read_csv(SENTIMENT_FILE)
    topic = pd.read_csv(TOPICS_FILE)

    merge_cols = [c for c in topic.columns if c not in sent.columns]
    key_cols   = ["city", "review_date", "property", "text_positive"]
    df = sent.merge(topic[key_cols + merge_cols], on=key_cols, how="left")

    df["city"] = df["city"].str.lower().str.strip()

    if sample:
        df = df.sample(min(sample, len(df)), random_state=42).reset_index(drop=True)
        log.info("Sampled %d rows.", len(df))
    else:
        log.info("Loaded %d rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# Temporal descriptive analysis
# ---------------------------------------------------------------------------

def run_temporal_analysis(df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df2026 = df[df["temporal_window"].isin(EVENT_WINDOWS)].copy()
    df2026["window"] = pd.Categorical(df2026["temporal_window"],
                                      categories=WINDOW_ORDER, ordered=True)

    cities = [c for c in [TARGET_CITY, CONTROL_CITY] if c in df2026["city"].unique()]

    _temporal_boxplot(
        df2026, metric="rating", ylabel="Rating (1–10)",
        title="Rating per finestra temporale (2026)",
        fname="temporal_rating_boxplot.png",
        cities=cities,
    )

    if "sentiment_score" in df2026.columns:
        _temporal_boxplot(
            df2026, metric="sentiment_score", ylabel="Sentiment score",
            title="Sentiment per finestra temporale (2026)",
            fname="temporal_sentiment_boxplot.png",
            cities=cities,
        )

    df_cag = df[df["city"] == TARGET_CITY].copy()
    _temporal_comparison_2026_vs_2025(df_cag)

    grp = (
        df2026
        .groupby(["city", "window"], observed=True)
        .agg(
            n=("rating", "count"),
            rating_mean=("rating", "mean"),
            rating_std=("rating", "std"),
            sentiment_mean=("sentiment_score", "mean")
                if "sentiment_score" in df2026.columns
                else ("rating", lambda x: float("nan")),
        )
        .round(3)
    )
    grp.to_csv(os.path.join(OUT_DIR, "temporal_summary.csv"))
    log.info("Saved temporal_summary.csv")
    log.info("\n%s", grp.to_string())


def _temporal_boxplot(df: pd.DataFrame, metric: str, ylabel: str,
                      title: str, fname: str, cities: list[str]) -> None:
    fig, axes = plt.subplots(1, len(cities), figsize=(5 * len(cities), 5),
                             sharey=True)
    if len(cities) == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        data_by_window = [
            df.loc[(df["city"] == city) & (df["window"] == w), metric].dropna().values
            for w in WINDOW_ORDER
        ]
        bp = ax.boxplot(
            data_by_window, patch_artist=True, notch=False,
            medianprops=dict(color="black", lw=2),
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=5),
        )
        for patch, w in zip(bp["boxes"], WINDOW_ORDER):
            patch.set_facecolor(PALETTE_WINDOW[w])
            patch.set_alpha(0.75)
            patch.set_edgecolor("#555555")
        ax.set_xticks(range(1, len(WINDOW_ORDER) + 1))
        ax.set_xticklabels([WINDOW_LABELS[w] for w in WINDOW_ORDER], fontsize=9)
        ax.set_title(city.title(), fontsize=11)
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        ax.grid(axis="y", color="#e3e7eb", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # Annotazioni: media (◆) e numero di osservazioni per finestra
        for j, (d, mean_val) in enumerate(
            zip(data_by_window, [np.mean(d) if len(d) else np.nan for d in data_by_window]), 1
        ):
            if len(d):
                ax.text(j, mean_val, f" {mean_val:.2f}", ha="left", va="center",
                        fontsize=8, color="#333333", fontweight="bold")
            ax.text(j, ax.get_ylim()[0], f"n={len(d)}", ha="center",
                    va="bottom", fontsize=7, color="#555555")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)
    log.info("Saved %s", fname)


def _temporal_comparison_2026_vs_2025(df_cag: pd.DataFrame) -> None:
    records = []
    for w_2026, w_2025 in zip(WINDOW_ORDER, BASELINE_ORDER):
        for year, window in [("2026", w_2026), ("2025", w_2025)]:
            sub = df_cag[df_cag["temporal_window"] == window]
            if len(sub) == 0:
                continue
            records.append({
                "window":    w_2026,
                "year":      year,
                "rating":    sub["rating"].mean(),
                "sentiment": sub["sentiment_score"].mean()
                             if "sentiment_score" in sub.columns else float("nan"),
                "n":         len(sub),
            })

    if not records:
        log.warning("No data for Cagliari 2026 vs 2025 comparison.")
        return

    comp      = pd.DataFrame(records)
    bar_width = 0.35
    x         = np.arange(len(WINDOW_ORDER))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel, title_suffix in [
        (ax1, "rating",    "Rating medio (1–10)",   "Rating"),
        (ax2, "sentiment", "Sentiment score medio",  "Sentiment"),
    ]:
        for year, offset, color in [
            ("2026", -bar_width / 2, PALETTE["year_2026"]),
            ("2025", +bar_width / 2, PALETTE["year_2025"]),
        ]:
            heights = [
                comp.loc[(comp["window"] == w) & (comp["year"] == year), metric]
                .values[0]
                if len(comp.loc[(comp["window"] == w) & (comp["year"] == year)]) else float("nan")
                for w in WINDOW_ORDER
            ]
            ns = [
                int(comp.loc[(comp["window"] == w) & (comp["year"] == year), "n"]
                    .values[0])
                if len(comp.loc[(comp["window"] == w) & (comp["year"] == year)]) else 0
                for w in WINDOW_ORDER
            ]
            bars = ax.bar(x + offset, heights, bar_width,
                          label=f"America's Cup {year}" if year == "2026" else f"Baseline {year}",
                          color=color, alpha=0.9, edgecolor="#555555", linewidth=0.6)
            # Margine verticale per le etichette, proporzionale al range dei valori
            valid = [v for v in comp[metric] if not np.isnan(v)]
            span = (max(valid) - min(valid)) if valid else 1.0
            pad  = max(span * 0.04, 0.02)
            for bar, h, n in zip(bars, heights, ns):
                if np.isnan(h):
                    continue
                if h >= 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            h + pad, f"{h:.2f}\n(n={n})", ha="center",
                            va="bottom", fontsize=7.5, color="#333333")
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            h - pad, f"{h:.2f}\n(n={n})", ha="center",
                            va="top", fontsize=7.5, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels([WINDOW_LABELS[w] for w in WINDOW_ORDER], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Cagliari — {title_suffix}: 2026 vs 2025")
        ax.legend(fontsize=9, loc="lower right" if metric == "rating" else "upper right")
        ax.grid(axis="y", color="#e3e7eb", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.axhline(0, color="#888888", linewidth=0.8)

        # Limiti y con margine extra per etichette sopra/sotto le barre
        valid = [v for v in comp[metric] if not np.isnan(v)]
        if valid:
            vmin, vmax = min(valid), max(valid)
            span = vmax - vmin if vmax != vmin else abs(vmax) or 1.0
            top    = vmax + span * 0.30 if vmax > 0 else span * 0.10
            bottom = vmin - span * 0.30 if vmin < 0 else 0
            ax.set_ylim(bottom, top)

    fig.suptitle("Cagliari: confronto America's Cup 2026 vs baseline 2025", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "temporal_cagliari_2026_vs_2025.png"), dpi=150)
    plt.close(fig)
    log.info("Saved temporal_cagliari_2026_vs_2025.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(sample: int | None = None) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data(sample=sample)
    run_temporal_analysis(df)
    log.info("Exploratory pipeline complete. Output → %s", OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploratory temporal analysis")
    parser.add_argument("--sample", type=int, default=None,
                        help="Analyse only N randomly sampled rows (for quick testing).")
    args = parser.parse_args()
    run_pipeline(sample=args.sample)
