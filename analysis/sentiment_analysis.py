"""
Sentiment and emotion analysis pipeline for tourist review data.

Sentiment: nlptown/bert-base-multilingual-uncased-sentiment is used for all
languages (Italian, English, unknown).  Falls back to TextBlob when nlptown
is unavailable.

Emotion: Feel-IT emotion model for Italian; DistilRoBERTa for English.
Language detection (langdetect) is still performed per-row for emotion dispatch.

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
HF_SENTIMENT_IT = "nlptown/bert-base-multilingual-uncased-sentiment"
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
# Sentiment — nlptown / TextBlob (Italian and multilingual)
# ---------------------------------------------------------------------------

# nlptown outputs 1-5 star labels; map to signed [-1, +1] score and polarity
_NLPTOWN_STAR_TO_LABEL = {
    "1 star":  "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}

_NLPTOWN_STAR_TO_SCORE = {
    "1 star":  -1.0,
    "2 stars": -0.5,
    "3 stars":  0.0,
    "4 stars":  0.5,
    "5 stars":  1.0,
}


def sentiment_nlptown(
    texts: list[str],
    pipeline,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Run nlptown multilingual sentiment pipeline in batches."""
    truncated = [t[:MAX_TOKEN_LEN * 4] if t else "" for t in texts]
    results = []
    for i in tqdm(range(0, len(truncated), batch_size), desc="nlptown sentiment", leave=False):
        batch = truncated[i : i + batch_size]
        batch = [t if t.strip() else " " for t in batch]
        preds = pipeline(batch, truncation=True, max_length=MAX_TOKEN_LEN)
        for pred in preds:
            if isinstance(pred, list):
                pred = pred[0]
            label_raw = pred["label"].lower()
            label = _NLPTOWN_STAR_TO_LABEL.get(label_raw, "neutral")
            score = _NLPTOWN_STAR_TO_SCORE.get(label_raw, 0.0)
            results.append({"sentiment_label": label, "sentiment_score": score})
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
# Feel-IT compatibility patches for transformers 5.x
# ---------------------------------------------------------------------------

_FEEL_IT_MODELS = {
    "MilaNLProc/feel-it-italian-sentiment",
    "MilaNLProc/feel-it-italian-emotion",
}

_PATCHES_DIR = Path(__file__).parent / "patches"


def _patch_camembert_tokenizer() -> None:
    """
    Monkey-patch CamemBERT tokenizer for transformers 5.x compatibility.

    In transformers 5.x the CamemBERT fast tokenizer tries to unpack each
    vocabulary entry as a 2-tuple (token, score), but SentencePiece produces
    3-tuples (token, score, type).  The patch normalises the vocab to 2-tuples
    and handles the case where the vocab arrives as a plain dict {token: id}.
    """
    try:
        from transformers.models.camembert import tokenization_camembert as _tc
        original_init = _tc.CamembertTokenizer.__init__

        def _patched_init(self, *args, vocab=None, **kwargs):
            if vocab is not None:
                if isinstance(vocab, dict):
                    vocab = list(vocab.items())
                else:
                    vocab = [(e[0], e[1]) for e in vocab]
            original_init(self, *args, vocab=vocab, **kwargs)

        if not getattr(_tc.CamembertTokenizer.__init__, "_feel_it_patched", False):
            _tc.CamembertTokenizer.__init__ = _patched_init
            _tc.CamembertTokenizer.__init__._feel_it_patched = True
    except Exception:
        pass


def _ensure_feel_it_tokenizer(model_id: str) -> None:
    """
    Copy the repo-patched tokenizer.json into the HuggingFace cache for
    *model_id* if the cached copy is missing the required 'type' field.

    This makes the fix venv- and cache-independent: the corrected file lives
    in analysis/patches/ and is applied on first use.
    """
    patch_file = _PATCHES_DIR / "feel_it_tokenizer.json"
    if not patch_file.exists():
        return

    try:
        import json
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(model_id, "tokenizer.json")
        with open(cached) as f:
            data = json.load(f)

        if data.get("model", {}).get("type") != "Unigram":
            import shutil
            shutil.copy(patch_file, cached)
            log.info("Applied Feel-IT tokenizer patch → %s", cached)
    except Exception:
        pass


def _apply_feel_it_patches(model_id: str) -> None:
    _patch_camembert_tokenizer()
    if model_id in _FEEL_IT_MODELS:
        _ensure_feel_it_tokenizer(model_id)


# ---------------------------------------------------------------------------
# Model loader — lazy, cached
# ---------------------------------------------------------------------------

_model_cache: dict[str, object] = {}


def _get_pipeline(model_id: str, task: str = "text-classification"):
    """Load a HuggingFace pipeline once and cache it in memory."""
    if model_id in _model_cache:
        return _model_cache[model_id]
    _apply_feel_it_patches(model_id)
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


def _nlptown_available() -> bool:
    """Return True if the nlptown sentiment model can be loaded."""
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
    nlptown_pipe,
    batch_size: int,
) -> list[dict]:
    """Run sentiment on all texts with nlptown, falling back to TextBlob."""
    if nlptown_pipe is not None:
        return sentiment_nlptown(texts, nlptown_pipe, batch_size)
    return sentiment_textblob(texts)


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

    nlptown_pipe = None
    try:
        nlptown_pipe = _get_pipeline(HF_SENTIMENT_IT)
        log.info("Using nlptown for all languages.")
    except Exception:
        log.warning("nlptown unavailable — falling back to TextBlob.")

    log.info("Running sentiment analysis on %d texts …", len(texts))
    results = _run_sentiment_batch(texts, nlptown_pipe, batch_size)

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
    city_col = "property_city" if "property_city" in df.columns else ("city" if "city" in df.columns else None)
    if city_col is None:
        log.warning("Column 'property_city'/'city' not found — skipping city aggregation.")
        return pd.DataFrame()

    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

    sentiment_agg = (
        df.groupby(city_col)["sentiment_score"]
        .agg(mean_sentiment="mean", review_count="count")
        .reset_index()
        .rename(columns={city_col: "property_city"})
    )

    emotion_dist = (
        df.groupby([city_col, "emotion_label"])
        .size()
        .reset_index(name="emotion_count")
        .pivot(index=city_col, columns="emotion_label", values="emotion_count")
        .fillna(0)
        .add_prefix("emotion_")
        .reset_index()
        .rename(columns={city_col: "property_city"})
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
