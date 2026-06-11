"""
Sentiment and emotion analysis pipeline for tourist review data.

Two-track NLP approach:
    - Italian text: Feel-IT (HuggingFace) for sentiment; Feel-IT emotion model
      for emotions.  Falls back to TextBlob when Feel-IT is unavailable.
    - English text: VADER for sentiment; DistilRoBERTa for emotions.

Language detection is performed per-row with langdetect.

Outputs
-------
data/processed/reviews_sentiment.csv   — full dataframe with new columns
results/sentiment_by_window.csv        — mean sentiment aggregated by temporal window
results/sentiment_by_city.csv          — mean sentiment aggregated by city
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from analysis.config import PROCESSED_DATA_DIR, RESULTS_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

INPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "reviews_processed.csv")
SENTIMENT_OUTPUT = os.path.join(PROCESSED_DATA_DIR, "reviews_sentiment.csv")
AGG_WINDOW_OUTPUT = os.path.join(RESULTS_DIR, "sentiment_by_window.csv")
AGG_CITY_OUTPUT = os.path.join(RESULTS_DIR, "sentiment_by_city.csv")

# Columns that contain the text to analyse (lemmatized)
TEXT_POSITIVE_COL = "text_positive_lemma"
TEXT_NEGATIVE_COL = "text_negative_lemma"

# HuggingFace model identifiers
HF_SENTIMENT_IT = "MilaNLProc/feel-it-italian-sentiment"
HF_EMOTION_IT = "MilaNLProc/feel-it-italian-emotion"
HF_EMOTION_EN = "j-hartmann/emotion-english-distilroberta-base"

# M2-friendly batch size (keeps peak RAM well under 8 GB)
DEFAULT_BATCH_SIZE = 8

# Maximum token length accepted by the transformer models
MAX_TOKEN_LEN = 512

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Return 'it', 'en', or 'unknown' for *text*.
    Uses langdetect; falls back to 'it' (project default) on any error.
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    try:
        from langdetect import detect, LangDetectException
        lang = detect(text)
        return lang if lang in ("it", "en") else "it"
    except Exception:
        return "it"


# ---------------------------------------------------------------------------
# Sentiment — VADER (English)
# ---------------------------------------------------------------------------

def _load_vader():
    """Load VADER analyser, downloading NLTK data if needed."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        log.info("Downloading VADER lexicon …")
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def _vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def sentiment_vader(texts: list[str], vader) -> list[dict]:
    """Return list of {sentiment_label, sentiment_score} dicts using VADER."""
    results = []
    for text in texts:
        if not text.strip():
            results.append({"sentiment_label": "neutral", "sentiment_score": 0.0})
            continue
        scores = vader.polarity_scores(text)
        compound = scores["compound"]
        results.append({
            "sentiment_label": _vader_label(compound),
            "sentiment_score": round(compound, 4),
        })
    return results


# ---------------------------------------------------------------------------
# Sentiment — Feel-IT / TextBlob (Italian)
# ---------------------------------------------------------------------------

def _hf_label_to_standard(label: str) -> str:
    """Map HuggingFace model output labels to positive/negative/neutral."""
    label = label.lower()
    if label in ("positive", "pos", "label_1", "1"):
        return "positive"
    if label in ("negative", "neg", "label_0", "0"):
        return "negative"
    return "neutral"


def sentiment_feel_it(
    texts: list[str],
    pipeline,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Run Feel-IT sentiment pipeline in batches."""
    # Truncate to avoid exceeding model max length
    truncated = [t[:MAX_TOKEN_LEN * 4] if t else "" for t in texts]
    results = []
    for i in tqdm(range(0, len(truncated), batch_size), desc="Feel-IT sentiment", leave=False):
        batch = truncated[i : i + batch_size]
        # Replace empty strings with a space so the model doesn't error
        batch = [t if t.strip() else " " for t in batch]
        preds = pipeline(batch, truncation=True, max_length=MAX_TOKEN_LEN)
        for pred in preds:
            label = _hf_label_to_standard(pred["label"])
            score = round(pred["score"], 4)
            # Convert score to signed: negative → negate
            signed = score if label == "positive" else (-score if label == "negative" else 0.0)
            results.append({"sentiment_label": label, "sentiment_score": signed})
    return results


def sentiment_textblob(texts: list[str]) -> list[dict]:
    """TextBlob-based sentiment fallback for Italian text."""
    from textblob import TextBlob
    results = []
    for text in texts:
        if not text.strip():
            results.append({"sentiment_label": "neutral", "sentiment_score": 0.0})
            continue
        polarity = TextBlob(text).sentiment.polarity
        label = "positive" if polarity > 0.05 else ("negative" if polarity < -0.05 else "neutral")
        results.append({"sentiment_label": label, "sentiment_score": round(polarity, 4)})
    return results


# ---------------------------------------------------------------------------
# Emotion — HuggingFace (Italian + English)
# ---------------------------------------------------------------------------

def emotion_hf(
    texts: list[str],
    pipeline,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Run a HuggingFace emotion pipeline in batches."""
    truncated = [t[:MAX_TOKEN_LEN * 4] if t else "" for t in texts]
    results = []
    for i in tqdm(range(0, len(truncated), batch_size), desc="Emotion model", leave=False):
        batch = truncated[i : i + batch_size]
        batch = [t if t.strip() else " " for t in batch]
        preds = pipeline(batch, truncation=True, max_length=MAX_TOKEN_LEN)
        for pred in preds:
            # Some pipelines return a list of dicts; take the top-scored entry
            if isinstance(pred, list):
                pred = max(pred, key=lambda x: x["score"])
            results.append({
                "emotion_label": pred["label"].lower(),
                "emotion_score": round(pred["score"], 4),
            })
    return results


# ---------------------------------------------------------------------------
# Model loader — lazy, cached
# ---------------------------------------------------------------------------

_model_cache: dict[str, object] = {}


def _get_pipeline(model_id: str, task: str = "text-classification"):
    """Load a HuggingFace pipeline once and cache it in memory."""
    if model_id in _model_cache:
        return _model_cache[model_id]
    from transformers import pipeline as hf_pipeline
    log.info("Loading model: %s …", model_id)
    pipe = hf_pipeline(
        task,
        model=model_id,
        device=-1,          # CPU — MPS support in transformers is still patchy
        top_k=1,
    )
    _model_cache[model_id] = pipe
    log.info("Model loaded: %s", model_id)
    return pipe


def _feel_it_available() -> bool:
    """Return True if the Feel-IT sentiment model can be loaded."""
    try:
        _get_pipeline(HF_SENTIMENT_IT)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-row dispatchers
# ---------------------------------------------------------------------------

def _run_sentiment_batch(
    texts: list[str],
    langs: list[str],
    vader,
    feel_it_pipe,
    batch_size: int,
) -> list[dict]:
    """Dispatch each text to the correct sentiment backend."""
    results: list[Optional[dict]] = [None] * len(texts)

    # Collect indices by language
    en_idx = [i for i, l in enumerate(langs) if l == "en"]
    it_idx = [i for i, l in enumerate(langs) if l != "en"]  # 'it' and 'unknown'

    if en_idx:
        en_texts = [texts[i] for i in en_idx]
        en_results = sentiment_vader(en_texts, vader)
        for pos, idx in enumerate(en_idx):
            results[idx] = en_results[pos]

    if it_idx:
        it_texts = [texts[i] for i in it_idx]
        if feel_it_pipe is not None:
            it_results = sentiment_feel_it(it_texts, feel_it_pipe, batch_size)
        else:
            it_results = sentiment_textblob(it_texts)
        for pos, idx in enumerate(it_idx):
            results[idx] = it_results[pos]

    return results  # type: ignore[return-value]


def _run_emotion_batch(
    texts: list[str],
    langs: list[str],
    emotion_it_pipe,
    emotion_en_pipe,
    batch_size: int,
) -> list[dict]:
    """Dispatch each text to the correct emotion model."""
    results: list[Optional[dict]] = [None] * len(texts)
    null_result = {"emotion_label": "unknown", "emotion_score": 0.0}

    en_idx = [i for i, l in enumerate(langs) if l == "en"]
    it_idx = [i for i, l in enumerate(langs) if l == "it"]
    unk_idx = [i for i, l in enumerate(langs) if l == "unknown"]

    for idx in unk_idx:
        results[idx] = null_result

    if en_idx and emotion_en_pipe is not None:
        en_results = emotion_hf([texts[i] for i in en_idx], emotion_en_pipe, batch_size)
        for pos, idx in enumerate(en_idx):
            results[idx] = en_results[pos]
    else:
        for idx in en_idx:
            results[idx] = null_result

    if it_idx and emotion_it_pipe is not None:
        it_results = emotion_hf([texts[i] for i in it_idx], emotion_it_pipe, batch_size)
        for pos, idx in enumerate(it_idx):
            results[idx] = it_results[pos]
    else:
        for idx in it_idx:
            results[idx] = null_result

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Step 1 — Load processed reviews
# ---------------------------------------------------------------------------

def load_processed(path: str = INPUT_FILE) -> pd.DataFrame:
    log.info("Loading processed reviews from '%s' …", path)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    log.info("Loaded %d rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 2 — Sentiment analysis
# ---------------------------------------------------------------------------

def run_sentiment(
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Add sentiment_label and sentiment_score columns to *df*.
    Uses text_positive_lemma as the primary text source.
    """
    texts = df[TEXT_POSITIVE_COL].fillna("").tolist()

    log.info("Detecting languages …")
    langs = [detect_language(t) for t in tqdm(texts, desc="Language detection")]

    vader = _load_vader()
    feel_it_pipe = None
    try:
        feel_it_pipe = _get_pipeline(HF_SENTIMENT_IT)
        log.info("Using Feel-IT for Italian sentiment.")
    except Exception:
        log.warning("Feel-IT unavailable — falling back to TextBlob for Italian.")

    log.info("Running sentiment analysis on %d texts …", len(texts))
    results = _run_sentiment_batch(texts, langs, vader, feel_it_pipe, batch_size)

    df["sentiment_label"] = [r["sentiment_label"] for r in results]
    df["sentiment_score"] = [r["sentiment_score"] for r in results]
    df["detected_language"] = langs
    return df


# ---------------------------------------------------------------------------
# Step 3 — Emotion analysis
# ---------------------------------------------------------------------------

def run_emotion(
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Add emotion_label and emotion_score columns to *df*.
    Uses text_positive_lemma as the primary text source.
    """
    texts = df[TEXT_POSITIVE_COL].fillna("").tolist()
    langs = df["detected_language"].tolist() if "detected_language" in df.columns \
        else [detect_language(t) for t in texts]

    emotion_it_pipe = None
    emotion_en_pipe = None
    try:
        emotion_it_pipe = _get_pipeline(HF_EMOTION_IT)
    except Exception:
        log.warning("Could not load Italian emotion model (%s).", HF_EMOTION_IT)
    try:
        emotion_en_pipe = _get_pipeline(HF_EMOTION_EN)
    except Exception:
        log.warning("Could not load English emotion model (%s).", HF_EMOTION_EN)

    log.info("Running emotion analysis on %d texts …", len(texts))
    results = _run_emotion_batch(texts, langs, emotion_it_pipe, emotion_en_pipe, batch_size)

    df["emotion_label"] = [r["emotion_label"] for r in results]
    df["emotion_score"] = [r["emotion_score"] for r in results]
    return df


# ---------------------------------------------------------------------------
# Step 4 — Aggregations
# ---------------------------------------------------------------------------

def aggregate_by_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean sentiment score and emotion distribution per temporal_window.
    Returns one row per window.
    """
    if "temporal_window" not in df.columns:
        log.warning("Column 'temporal_window' not found — skipping window aggregation.")
        return pd.DataFrame()

    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

    sentiment_agg = (
        df.groupby("temporal_window")["sentiment_score"]
        .agg(mean_sentiment="mean", review_count="count")
        .reset_index()
    )

    emotion_dist = (
        df.groupby(["temporal_window", "emotion_label"])
        .size()
        .reset_index(name="emotion_count")
        .pivot(index="temporal_window", columns="emotion_label", values="emotion_count")
        .fillna(0)
        .add_prefix("emotion_")
        .reset_index()
    )

    return sentiment_agg.merge(emotion_dist, on="temporal_window", how="left")


def aggregate_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean sentiment score and emotion distribution per property_city.
    Returns one row per city.
    """
    if "property_city" not in df.columns:
        log.warning("Column 'property_city' not found — skipping city aggregation.")
        return pd.DataFrame()

    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

    sentiment_agg = (
        df.groupby("property_city")["sentiment_score"]
        .agg(mean_sentiment="mean", review_count="count")
        .reset_index()
    )

    emotion_dist = (
        df.groupby(["property_city", "emotion_label"])
        .size()
        .reset_index(name="emotion_count")
        .pivot(index="property_city", columns="emotion_label", values="emotion_count")
        .fillna(0)
        .add_prefix("emotion_")
        .reset_index()
    )

    return sentiment_agg.merge(emotion_dist, on="property_city", how="left")


# ---------------------------------------------------------------------------
# Step 5 — Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    df: pd.DataFrame,
    agg_window: pd.DataFrame,
    agg_city: pd.DataFrame,
) -> None:
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    df.to_csv(SENTIMENT_OUTPUT, index=False)
    log.info("Saved full dataframe → %s", SENTIMENT_OUTPUT)

    if not agg_window.empty:
        agg_window.to_csv(AGG_WINDOW_OUTPUT, index=False)
        log.info("Saved window aggregation → %s", AGG_WINDOW_OUTPUT)

    if not agg_city.empty:
        agg_city.to_csv(AGG_CITY_OUTPUT, index=False)
        log.info("Saved city aggregation → %s", AGG_CITY_OUTPUT)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str = INPUT_FILE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Execute the full sentiment + emotion pipeline.

    Parameters
    ----------
    input_path:
        Path to the preprocessed CSV produced by preprocessing.py.
    batch_size:
        Number of texts processed per HuggingFace forward pass.
        Default of 8 is tuned for Apple M2 with 8 GB RAM.
    sample_size:
        If set, run only on the first *sample_size* rows.  Use this to
        verify the pipeline quickly before processing the full dataset.
    """
    df = load_processed(input_path)

    if df.empty:
        log.warning("Input dataframe is empty — aborting.")
        return df

    if sample_size is not None:
        log.info("Sample mode: using first %d rows.", sample_size)
        df = df.head(sample_size).copy()

    # Ensure the lemma column exists; fall back to raw positive text
    if TEXT_POSITIVE_COL not in df.columns:
        fallback = "text_positive"
        log.warning(
            "'%s' not found — falling back to '%s'.", TEXT_POSITIVE_COL, fallback
        )
        df[TEXT_POSITIVE_COL] = df.get(fallback, pd.Series(dtype=str))

    df = run_sentiment(df, batch_size)
    df = run_emotion(df, batch_size)

    agg_window = aggregate_by_window(df)
    agg_city = aggregate_by_city(df)

    save_outputs(df, agg_window, agg_city)

    log.info("Pipeline complete.  Shape: %s", df.shape)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run sentiment analysis pipeline.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only the first N rows (for testing).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"HuggingFace batch size (default: {DEFAULT_BATCH_SIZE}).")
    args = parser.parse_args()

    run_pipeline(sample_size=args.sample, batch_size=args.batch_size)
