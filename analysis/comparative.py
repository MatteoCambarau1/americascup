"""
Comparative analysis module — final integration step.

Reads the outputs of preprocessing, sentiment analysis, and topic modelling
and produces all cross-dimensional comparisons needed to assess the tourism
impact of the America's Cup 2026 in Cagliari.

Analyses produced
-----------------
1. Volumetric  — review counts by window / city, YoY % change
2. Sentiment × window  — 2026 vs 2025 baseline, per city
3. Sentiment × city    — Cagliari vs Olbia within 2026
4. Topics × window     — dominant topics 2026 vs 2025
5. Topics × city       — topic distribution Cagliari vs Olbia
6. Summary report      — plain-text narrative of key findings

Outputs (all in results/)
-------------------------
volumetric_analysis.csv
sentiment_comparison_windows.csv
sentiment_comparison_cities.csv
topic_comparison_windows.csv
topic_comparison_cities.csv
summary_report.txt
"""

from __future__ import annotations

import logging
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from analysis.config import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    TARGET_CITY,
    CONTROL_CITY,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

PREPROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "reviews_processed.csv")
SENTIMENT_FILE = os.path.join(PROCESSED_DATA_DIR, "reviews_sentiment.csv")
TOPICS_FILE = os.path.join(PROCESSED_DATA_DIR, "reviews_topics.csv")

OUT_VOLUMETRIC = os.path.join(RESULTS_DIR, "volumetric_analysis.csv")
OUT_SENT_WINDOW = os.path.join(RESULTS_DIR, "sentiment_comparison_windows.csv")
OUT_SENT_CITY = os.path.join(RESULTS_DIR, "sentiment_comparison_cities.csv")
OUT_TOPIC_WINDOW = os.path.join(RESULTS_DIR, "topic_comparison_windows.csv")
OUT_TOPIC_CITY = os.path.join(RESULTS_DIR, "topic_comparison_cities.csv")
OUT_SUMMARY = os.path.join(RESULTS_DIR, "summary_report.txt")

# Canonical window pairs: (event_2026_label, baseline_2025_label)
WINDOW_PAIRS = [
    ("pre", "baseline_pre"),
    ("during", "baseline_during"),
    ("post", "baseline_post"),
]

# Shorthand for all 6 window labels
EVENT_WINDOWS = {"pre", "during", "post"}
BASELINE_WINDOWS = {"baseline_pre", "baseline_during", "baseline_post"}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved → %s  (%d rows)", path, len(df))


def _pct_change(new: float, old: float) -> Optional[float]:
    """Return percentage change from *old* to *new*, or None when old is 0."""
    if old == 0:
        return None
    return round((new - old) / abs(old) * 100, 2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load and merge the three processed files on a common row index.

    The merge is left-join on positional index so rows always align even
    when some downstream files cover only a subset of columns.
    Numeric columns (rating, sentiment_score, etc.) are cast after merge.
    """
    steps = [
        ("preprocessed", PREPROCESSED_FILE),
        ("sentiment", SENTIMENT_FILE),
        ("topics", TOPICS_FILE),
    ]

    frames: dict[str, pd.DataFrame] = {}
    for name, path in tqdm(steps, desc="Loading files"):
        if not os.path.exists(path):
            log.warning("File not found: '%s' — continuing without it.", path)
            continue
        frames[name] = pd.read_csv(path, dtype=str, low_memory=False)
        log.info("  %-14s %d rows × %d cols", name, *frames[name].shape)

    if not frames:
        raise FileNotFoundError("No input files found in data/processed/.")

    # Start with the widest base; merge additional columns from other files
    base_name = "preprocessed" if "preprocessed" in frames else next(iter(frames))
    df = frames[base_name].copy()

    for name, frame in frames.items():
        if name == base_name:
            continue
        new_cols = [c for c in frame.columns if c not in df.columns]
        if new_cols:
            df = pd.concat([df, frame[new_cols]], axis=1)

    # Cast numeric columns
    for col in ("rating", "sentiment_score", "property_stars", "nights_stayed"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("Merged dataframe: %d rows × %d cols", *df.shape)
    return df


# ---------------------------------------------------------------------------
# 1. Volumetric analysis
# ---------------------------------------------------------------------------

def volumetric_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Review counts per (temporal_window, property_city) with YoY % change.

    For each event window (pre/during/post) the count is compared against
    the corresponding baseline window from 2025.
    """
    if "temporal_window" not in df.columns:
        log.warning("'temporal_window' not found — skipping volumetric analysis.")
        return pd.DataFrame()

    group_cols = ["temporal_window", "property_city"]
    available = [c for c in group_cols if c in df.columns]

    counts = (
        df.groupby(available)
        .size()
        .reset_index(name="review_count")
    )

    if "property_city" not in available:
        # No city column — just window-level counts
        counts["pct_change_vs_baseline"] = None
        return counts

    rows = []
    cities = counts["property_city"].unique()

    for city in tqdm(cities, desc="Volumetric analysis"):
        city_df = counts[counts["property_city"] == city]

        for event_w, baseline_w in WINDOW_PAIRS:
            event_row = city_df[city_df["temporal_window"] == event_w]
            base_row = city_df[city_df["temporal_window"] == baseline_w]

            event_count = int(event_row["review_count"].values[0]) if not event_row.empty else 0
            base_count = int(base_row["review_count"].values[0]) if not base_row.empty else 0

            rows.append({
                "property_city": city,
                "window_2026": event_w,
                "window_2025": baseline_w,
                "reviews_2026": event_count,
                "reviews_2025": base_count,
                "pct_change": _pct_change(event_count, base_count),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Sentiment × temporal window (2026 vs 2025 baseline)
# ---------------------------------------------------------------------------

def sentiment_by_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean sentiment score and positive-review share per window, split by year
    cohort (2026 vs 2025 baseline).  Also computes % change between cohorts.
    """
    needed = {"temporal_window", "sentiment_score", "sentiment_label"}
    missing = needed - set(df.columns)
    if missing:
        log.warning("Skipping sentiment×window — missing columns: %s", missing)
        return pd.DataFrame()

    def _agg(sub: pd.DataFrame) -> dict:
        total = len(sub)
        pos = (sub["sentiment_label"] == "positive").sum()
        neg = (sub["sentiment_label"] == "negative").sum()
        return {
            "review_count": total,
            "mean_sentiment": round(sub["sentiment_score"].mean(), 4),
            "pct_positive": round(pos / total * 100, 2) if total else 0.0,
            "pct_negative": round(neg / total * 100, 2) if total else 0.0,
        }

    rows = []
    cities = df["property_city"].unique() if "property_city" in df.columns else [None]

    for city in tqdm(cities, desc="Sentiment × window"):
        city_df = df if city is None else df[df["property_city"] == city]

        for event_w, baseline_w in WINDOW_PAIRS:
            e_sub = city_df[city_df["temporal_window"] == event_w]
            b_sub = city_df[city_df["temporal_window"] == baseline_w]

            e_stats = _agg(e_sub) if not e_sub.empty else {}
            b_stats = _agg(b_sub) if not b_sub.empty else {}

            row: dict = {
                "property_city": city,
                "window": event_w,
            }
            for k, v in e_stats.items():
                row[f"{k}_2026"] = v
            for k, v in b_stats.items():
                row[f"{k}_2025"] = v

            row["pct_change_mean_sentiment"] = _pct_change(
                e_stats.get("mean_sentiment", 0),
                b_stats.get("mean_sentiment", 0),
            )
            row["pct_change_positive"] = _pct_change(
                e_stats.get("pct_positive", 0),
                b_stats.get("pct_positive", 0),
            )
            row["pct_change_negative"] = _pct_change(
                e_stats.get("pct_negative", 0),
                b_stats.get("pct_negative", 0),
            )
            rows.append(row)

    result = pd.DataFrame(rows)

    # Emotion distribution per window (wide pivot)
    if "emotion_label" in df.columns:
        emotion_dist = (
            df[df["temporal_window"].isin(EVENT_WINDOWS | BASELINE_WINDOWS)]
            .groupby(["temporal_window", "emotion_label"])
            .size()
            .reset_index(name="count")
            .pivot(index="temporal_window", columns="emotion_label", values="count")
            .fillna(0).astype(int)
            .add_prefix("emotion_")
            .reset_index()
        )
        emotion_dist.columns.name = None
        result = result.merge(emotion_dist, left_on="window", right_on="temporal_window", how="left")
        result.drop(columns=["temporal_window"], errors="ignore", inplace=True)

    return result


# ---------------------------------------------------------------------------
# 3. Sentiment × city (Cagliari vs Olbia, 2026 only)
# ---------------------------------------------------------------------------

def sentiment_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean sentiment and label distribution for TARGET vs CONTROL city,
    restricted to 2026 event windows only.
    """
    needed = {"property_city", "sentiment_score", "sentiment_label", "temporal_window"}
    if missing := needed - set(df.columns):
        log.warning("Skipping sentiment×city — missing columns: %s", missing)
        return pd.DataFrame()

    df_2026 = df[df["temporal_window"].isin(EVENT_WINDOWS)].copy()

    rows = []
    for window in tqdm(["pre", "during", "post"], desc="Sentiment × city"):
        w_df = df_2026[df_2026["temporal_window"] == window]

        for city in [TARGET_CITY, CONTROL_CITY]:
            sub = w_df[w_df["property_city"] == city]
            if sub.empty:
                continue
            total = len(sub)
            rows.append({
                "temporal_window": window,
                "property_city": city,
                "review_count": total,
                "mean_sentiment": round(sub["sentiment_score"].mean(), 4),
                "pct_positive": round((sub["sentiment_label"] == "positive").sum() / total * 100, 2),
                "pct_negative": round((sub["sentiment_label"] == "negative").sum() / total * 100, 2),
                "pct_neutral":  round((sub["sentiment_label"] == "neutral").sum()  / total * 100, 2),
            })

    result = pd.DataFrame(rows)

    # Add % difference TARGET vs CONTROL per window
    diff_rows = []
    for window, grp in result.groupby("temporal_window"):
        target = grp[grp["property_city"] == TARGET_CITY]
        control = grp[grp["property_city"] == CONTROL_CITY]
        if target.empty or control.empty:
            continue
        t = target.iloc[0]
        c = control.iloc[0]
        diff_rows.append({
            "temporal_window": window,
            "diff_mean_sentiment": round(t["mean_sentiment"] - c["mean_sentiment"], 4),
            "diff_pct_positive": _pct_change(t["pct_positive"], c["pct_positive"]),
            "diff_pct_negative": _pct_change(t["pct_negative"], c["pct_negative"]),
        })

    if diff_rows:
        result = result.merge(pd.DataFrame(diff_rows), on="temporal_window", how="left")

    return result


# ---------------------------------------------------------------------------
# 4. Topics × window (2026 vs baseline 2025)
# ---------------------------------------------------------------------------

def topic_comparison_by_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dominant LDA topic distribution per window, comparing 2026 vs 2025.
    Returns counts of each topic label within each window.
    """
    lda_col = "lda_topic_positive"
    if lda_col not in df.columns or "temporal_window" not in df.columns:
        log.warning("Skipping topic×window — '%s' or 'temporal_window' not found.", lda_col)
        return pd.DataFrame()

    rows = []
    for event_w, baseline_w in tqdm(WINDOW_PAIRS, desc="Topics × window"):
        for window, year_label in [(event_w, "2026"), (baseline_w, "2025")]:
            sub = df[df["temporal_window"] == window]
            if sub.empty:
                continue
            counts = sub[lda_col].value_counts().reset_index()
            counts.columns = ["topic", "count"]
            counts["window"] = event_w       # use canonical name for merging
            counts["year"] = year_label
            counts["pct"] = round(counts["count"] / counts["count"].sum() * 100, 2)
            rows.append(counts)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 5. Topics × city (Cagliari vs Olbia, 2026)
# ---------------------------------------------------------------------------

def topic_comparison_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    LDA topic distribution per city, restricted to 2026 event windows.
    """
    lda_col = "lda_topic_positive"
    if lda_col not in df.columns or "property_city" not in df.columns:
        log.warning("Skipping topic×city — '%s' or 'property_city' not found.", lda_col)
        return pd.DataFrame()

    df_2026 = df[df["temporal_window"].isin(EVENT_WINDOWS)]

    rows = []
    for city in tqdm([TARGET_CITY, CONTROL_CITY], desc="Topics × city"):
        sub = df_2026[df_2026["property_city"] == city]
        if sub.empty:
            continue
        counts = sub[lda_col].value_counts().reset_index()
        counts.columns = ["topic", "count"]
        counts["property_city"] = city
        counts["pct"] = round(counts["count"] / counts["count"].sum() * 100, 2)
        rows.append(counts)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 6. Summary report
# ---------------------------------------------------------------------------

def _fmt(val, suffix: str = "") -> str:
    """Format a numeric value for the text report, handling None gracefully."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "n/a"
    return f"{val:+.1f}{suffix}" if suffix else f"{val:.4f}"


def build_summary_report(
    df: pd.DataFrame,
    vol: pd.DataFrame,
    sent_win: pd.DataFrame,
    sent_city: pd.DataFrame,
    topic_win: pd.DataFrame,
    topic_city: pd.DataFrame,
) -> str:
    """
    Produce a plain-text narrative summarising the key findings.
    Returns the report as a string (also written to disk by run_pipeline).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_reviews = len(df)
    target_reviews = len(df[df.get("property_city", pd.Series()) == TARGET_CITY]) \
        if "property_city" in df.columns else "n/a"

    lines = [
        "=" * 70,
        "  AMERICA'S CUP 2026 — TOURIST REVIEW IMPACT ANALYSIS",
        f"  Report generated: {now}",
        "=" * 70,
        "",
        f"Total reviews analysed : {total_reviews:,}",
        f"Reviews for {TARGET_CITY:<12}: {target_reviews:,}" if isinstance(target_reviews, int) else "",
        "",
    ]

    # --- Volumetric ---
    lines += ["─" * 70, "1. VOLUMETRIC ANALYSIS", "─" * 70]
    if not vol.empty and "window_2026" in vol.columns:
        for _, row in vol.iterrows():
            city = row.get("property_city", "all")
            lines.append(
                f"  {city:<12}  {row['window_2026']:<8}  "
                f"2026: {int(row.get('reviews_2026', 0)):>4}  "
                f"2025: {int(row.get('reviews_2025', 0)):>4}  "
                f"Δ: {_fmt(row.get('pct_change'), '%')}"
            )
    else:
        lines.append("  (data not available)")

    # --- Sentiment × window ---
    lines += ["", "─" * 70, "2. SENTIMENT — 2026 vs 2025 BASELINE", "─" * 70]
    if not sent_win.empty:
        for _, row in sent_win.iterrows():
            city = row.get("property_city") or "all"
            window = row.get("window") or "?"
            ms26 = row.get("mean_sentiment_2026")
            ms25 = row.get("mean_sentiment_2025")
            delta = row.get("pct_change_mean_sentiment")
            lines.append(
                f"  {city:<12}  {window:<8}  "
                f"mean sent 2026: {_fmt(ms26)}  "
                f"2025: {_fmt(ms25)}  "
                f"Δ: {_fmt(delta, '%')}"
            )
    else:
        lines.append("  (data not available)")

    # --- Sentiment × city ---
    lines += ["", "─" * 70, "3. SENTIMENT — CAGLIARI vs OLBIA (2026)", "─" * 70]
    if not sent_city.empty:
        for _, row in sent_city.iterrows():
            lines.append(
                f"  {row.get('temporal_window','?'):<8}  "
                f"{row.get('property_city','?'):<12}  "
                f"mean: {_fmt(row.get('mean_sentiment'))}  "
                f"pos: {row.get('pct_positive','?')}%  "
                f"neg: {row.get('pct_negative','?')}%"
            )
    else:
        lines.append("  (data not available)")

    # --- Topic × window ---
    lines += ["", "─" * 70, "4. DOMINANT TOPICS — 2026 vs 2025", "─" * 70]
    if not topic_win.empty:
        for window in ["pre", "during", "post"]:
            sub = topic_win[topic_win["window"] == window]
            if sub.empty:
                continue
            lines.append(f"  Window: {window}")
            for year in ["2026", "2025"]:
                top3 = (
                    sub[sub["year"] == year]
                    .nlargest(3, "count")[["topic", "pct"]]
                    .apply(lambda r: f"topic {int(r['topic'])} ({r['pct']}%)", axis=1)
                    .tolist()
                )
                lines.append(f"    {year}: {', '.join(top3) if top3 else 'n/a'}")
    else:
        lines.append("  (data not available)")

    # --- Topic × city ---
    lines += ["", "─" * 70, "5. DOMINANT TOPICS — CAGLIARI vs OLBIA (2026)", "─" * 70]
    if not topic_city.empty:
        for city in [TARGET_CITY, CONTROL_CITY]:
            sub = topic_city[topic_city["property_city"] == city]
            top3 = (
                sub.nlargest(3, "count")[["topic", "pct"]]
                .apply(lambda r: f"topic {int(r['topic'])} ({r['pct']}%)", axis=1)
                .tolist()
            ) if not sub.empty else ["n/a"]
            lines.append(f"  {city:<12}: {', '.join(top3)}")
    else:
        lines.append("  (data not available)")

    lines += ["", "=" * 70, "END OF REPORT", "=" * 70]
    return "\n".join(line for line in lines if line is not None)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> dict[str, pd.DataFrame]:
    """
    Execute all comparative analyses and write results to results/.

    Returns a dict mapping output names to their DataFrames for
    interactive inspection (e.g. in a Jupyter notebook).
    """
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    df = load_data()
    # Normalise column names: scraper outputs 'city', schema expects 'property_city'
    if "city" in df.columns and "property_city" not in df.columns:
        df["property_city"] = df["city"].str.title()

    steps = [
        ("volumetric",       volumetric_analysis),
        ("sentiment_window", sentiment_by_window),
        ("sentiment_city",   sentiment_by_city),
        ("topic_window",     topic_comparison_by_window),
        ("topic_city",       topic_comparison_by_city),
    ]

    results: dict[str, pd.DataFrame] = {}
    for name, fn in tqdm(steps, desc="Comparative analyses"):
        log.info("Running: %s …", name)
        results[name] = fn(df)

    # Save CSVs
    output_map = {
        "volumetric":       OUT_VOLUMETRIC,
        "sentiment_window": OUT_SENT_WINDOW,
        "sentiment_city":   OUT_SENT_CITY,
        "topic_window":     OUT_TOPIC_WINDOW,
        "topic_city":       OUT_TOPIC_CITY,
    }
    for name, path in output_map.items():
        if name in results and not results[name].empty:
            _save(results[name], path)

    # Build and save text report
    report = build_summary_report(
        df=df,
        vol=results.get("volumetric", pd.DataFrame()),
        sent_win=results.get("sentiment_window", pd.DataFrame()),
        sent_city=results.get("sentiment_city", pd.DataFrame()),
        topic_win=results.get("topic_window", pd.DataFrame()),
        topic_city=results.get("topic_city", pd.DataFrame()),
    )
    Path(OUT_SUMMARY).write_text(report, encoding="utf-8")
    log.info("Saved → %s", OUT_SUMMARY)

    # Print the summary to console as well
    print("\n" + report)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
