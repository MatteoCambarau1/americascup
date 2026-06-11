"""
NLP model validation module.

Validates the three models used in the pipeline against public benchmark
datasets before running inference on the full review corpus.

Models validated
----------------
1. Feel-IT sentiment  (MilaNLProc/feel-it-italian-sentiment)
   Benchmark: cardiffnlp/tweet_sentiment_multilingual — Italian split

2. VADER sentiment    (nltk VADER)
   Benchmark: cardiffnlp/tweet_sentiment_multilingual — English split

3. Feel-IT emotion    (MilaNLProc/feel-it-italian-emotion)
   Benchmark: MilaNLProc/de-it-fr-multilingual-go-emotions — Italian split

Each validation run:
  - Samples up to MAX_SAMPLES examples (reproducible via RANDOM_STATE)
  - Runs batch inference (BATCH_SIZE=8, CPU — safe for M2 8 GB)
  - Normalises labels so dataset and model outputs are comparable
  - Prints a classification report
  - Saves a confusion-matrix PNG to results/validation/
  - Appends a summary row to results/validation/validation_report.csv
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless — no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from analysis.config import RESULTS_DIR

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
# Constants
# ---------------------------------------------------------------------------

VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")
REPORT_PATH = os.path.join(VALIDATION_DIR, "validation_report.csv")

MAX_SAMPLES = 500
BATCH_SIZE = 8
RANDOM_STATE = 42

# HuggingFace model IDs
MODEL_FEEL_IT_SENTIMENT = "MilaNLProc/feel-it-italian-sentiment"
MODEL_FEEL_IT_EMOTION = "MilaNLProc/feel-it-italian-emotion"

# HuggingFace dataset IDs
DS_TWEET_MULTILINGUAL = "cardiffnlp/tweet_sentiment_multilingual"
DS_GO_EMOTIONS_MULTILINGUAL = "MilaNLProc/de-it-fr-multilingual-go-emotions"

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    Path(VALIDATION_DIR).mkdir(parents=True, exist_ok=True)


def _save_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    filename: str,
) -> str:
    """Render *cm* as a heatmap PNG and return the saved path."""
    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(4, len(labels))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    out_path = os.path.join(VALIDATION_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", out_path)
    return out_path


def _append_to_report(row: dict) -> None:
    """Append one summary row to the CSV report (creates file if absent)."""
    df_row = pd.DataFrame([row])
    if os.path.exists(REPORT_PATH):
        df_row.to_csv(REPORT_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(REPORT_PATH, index=False)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_tweet_sentiment(lang: str, max_samples: int = MAX_SAMPLES) -> pd.DataFrame:
    """
    Load the Cardiff tweet-sentiment dataset for *lang* ('it' or 'en').

    Label mapping (dataset integers → strings):
        0 → negative, 1 → neutral, 2 → positive
    """
    from datasets import load_dataset

    log.info("Loading %s — language: %s …", DS_TWEET_MULTILINGUAL, lang)
    ds = load_dataset(DS_TWEET_MULTILINGUAL, lang, split="test")

    df = ds.to_pandas()[["text", "label"]].dropna()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    df["label_str"] = df["label"].map(label_map)
    df = df.dropna(subset=["label_str"])

    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=RANDOM_STATE)

    log.info("  %d examples loaded.", len(df))
    return df.reset_index(drop=True)


def _load_go_emotions_italian(max_samples: int = MAX_SAMPLES) -> pd.DataFrame:
    """
    Load the multilingual GoEmotions dataset, Italian split.

    The dataset stores labels as integers mapped to emotion strings.
    We keep only examples whose label appears in the Feel-IT emotion
    label set: {joy, sadness, anger, fear, surprise, disgust}.
    """
    from datasets import load_dataset

    log.info("Loading %s — language: it …", DS_GO_EMOTIONS_MULTILINGUAL)
    ds = load_dataset(DS_GO_EMOTIONS_MULTILINGUAL, split="test")

    df = ds.to_pandas()

    # Filter to Italian rows if a language column is present
    if "language" in df.columns:
        df = df[df["language"] == "it"]
    elif "lang" in df.columns:
        df = df[df["lang"] == "it"]

    # The label column may be an integer id; convert via dataset features
    feel_it_emotions = {"joy", "sadness", "anger", "fear", "surprise", "disgust"}

    # Try to resolve label names from dataset metadata
    if hasattr(ds, "features") and "label" in ds.features:
        feature = ds.features["label"]
        if hasattr(feature, "names"):
            id2label = {i: n.lower() for i, n in enumerate(feature.names)}
            df["label_str"] = df["label"].map(id2label)
        else:
            df["label_str"] = df["label"].astype(str).str.lower()
    else:
        df["label_str"] = df["label"].astype(str).str.lower()

    # Keep only emotions the Feel-IT model was trained on
    df = df[df["label_str"].isin(feel_it_emotions)].dropna(subset=["text", "label_str"])

    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=RANDOM_STATE)

    log.info("  %d examples loaded (feel-it emotions only).", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _hf_predict_batched(
    texts: list[str],
    pipeline,
    batch_size: int = BATCH_SIZE,
) -> list[str]:
    """
    Run a HuggingFace text-classification pipeline in batches.
    Returns a list of lower-cased label strings.
    """
    predictions: list[str] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Inference", leave=False):
        batch = texts[i : i + batch_size]
        batch = [t if isinstance(t, str) and t.strip() else " " for t in batch]
        preds = pipeline(batch, truncation=True, max_length=512)
        for pred in preds:
            if isinstance(pred, list):
                pred = max(pred, key=lambda x: x["score"])
            predictions.append(pred["label"].lower())
    return predictions


def _load_hf_pipeline(model_id: str):
    """Load a HuggingFace classification pipeline on CPU."""
    from transformers import pipeline as hf_pipeline
    log.info("Loading model: %s …", model_id)
    return hf_pipeline(
        "text-classification",
        model=model_id,
        device=-1,      # CPU — MPS support is unstable in transformers
        top_k=1,
    )


# ---------------------------------------------------------------------------
# Label normalisers
# ---------------------------------------------------------------------------

def _normalise_feel_it_sentiment(label: str) -> str:
    """Map Feel-IT sentiment output to the three-class schema."""
    mapping = {
        "positive": "positive",
        "negative": "negative",
        "label_0": "negative",
        "label_1": "positive",
    }
    return mapping.get(label.lower(), "neutral")


def _normalise_vader(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _normalise_feel_it_emotion(label: str) -> str:
    """Lower-case and strip the Feel-IT emotion label."""
    return label.lower().strip()


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------

def _print_results(model_name: str, true: list, pred: list, labels: list[str]) -> None:
    acc = accuracy_score(true, pred)
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"  Accuracy: {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"{'=' * 60}")
    print(classification_report(true, pred, labels=labels, zero_division=0))


def _run_validation(
    model_name: str,
    true_labels: list[str],
    pred_labels: list[str],
    label_set: list[str],
    cm_filename: str,
) -> dict:
    """
    Compute metrics, render confusion matrix, append to report.
    Returns a summary dict for the caller.
    """
    acc = accuracy_score(true_labels, pred_labels)
    report = classification_report(
        true_labels, pred_labels, labels=label_set, zero_division=0, output_dict=True
    )

    _print_results(model_name, true_labels, pred_labels, label_set)

    cm = confusion_matrix(true_labels, pred_labels, labels=label_set)
    _save_confusion_matrix(
        cm,
        labels=label_set,
        title=f"{model_name}\nConfusion Matrix",
        filename=cm_filename,
    )

    summary = {
        "model": model_name,
        "n_samples": len(true_labels),
        "accuracy": round(acc, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    }
    _append_to_report(summary)
    return summary


# ---------------------------------------------------------------------------
# 1. Validate Feel-IT sentiment (Italian)
# ---------------------------------------------------------------------------

def validate_feel_it_sentiment(
    max_samples: int = MAX_SAMPLES,
    batch_size: int = BATCH_SIZE,
) -> Optional[dict]:
    """Validate Feel-IT Italian sentiment against the Cardiff tweet dataset."""
    log.info("─── Validating: Feel-IT Sentiment ───")
    try:
        df = _load_tweet_sentiment("it", max_samples)
        pipe = _load_hf_pipeline(MODEL_FEEL_IT_SENTIMENT)

        raw_preds = _hf_predict_batched(df["text"].tolist(), pipe, batch_size)
        pred_labels = [_normalise_feel_it_sentiment(p) for p in raw_preds]
        true_labels = df["label_str"].tolist()

        return _run_validation(
            model_name="Feel-IT Sentiment (Italian)",
            true_labels=true_labels,
            pred_labels=pred_labels,
            label_set=["positive", "negative", "neutral"],
            cm_filename="cm_feel_it_sentiment.png",
        )
    except Exception as exc:
        log.error("Feel-IT sentiment validation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 2. Validate VADER sentiment (English)
# ---------------------------------------------------------------------------

def validate_vader_sentiment(
    max_samples: int = MAX_SAMPLES,
    batch_size: int = BATCH_SIZE,   # unused for VADER, kept for API consistency
) -> Optional[dict]:
    """Validate VADER against the Cardiff English tweet-sentiment dataset."""
    log.info("─── Validating: VADER Sentiment ───")
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        for res in ("vader_lexicon",):
            try:
                nltk.data.find(f"sentiment/{res}.zip")
            except LookupError:
                nltk.download(res, quiet=True)

        df = _load_tweet_sentiment("en", max_samples)
        sid = SentimentIntensityAnalyzer()

        pred_labels: list[str] = []
        for text in tqdm(df["text"].tolist(), desc="  VADER inference", leave=False):
            compound = sid.polarity_scores(str(text))["compound"]
            pred_labels.append(_normalise_vader(compound))

        true_labels = df["label_str"].tolist()

        return _run_validation(
            model_name="VADER Sentiment (English)",
            true_labels=true_labels,
            pred_labels=pred_labels,
            label_set=["positive", "negative", "neutral"],
            cm_filename="cm_vader_sentiment.png",
        )
    except Exception as exc:
        log.error("VADER sentiment validation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 3. Validate Feel-IT emotion (Italian)
# ---------------------------------------------------------------------------

def validate_feel_it_emotion(
    max_samples: int = MAX_SAMPLES,
    batch_size: int = BATCH_SIZE,
) -> Optional[dict]:
    """Validate Feel-IT emotion against the multilingual GoEmotions dataset."""
    log.info("─── Validating: Feel-IT Emotion ───")
    try:
        df = _load_go_emotions_italian(max_samples)

        if df.empty:
            log.warning("No Italian emotion examples found — skipping.")
            return None

        pipe = _load_hf_pipeline(MODEL_FEEL_IT_EMOTION)

        raw_preds = _hf_predict_batched(df["text"].tolist(), pipe, batch_size)
        pred_labels = [_normalise_feel_it_emotion(p) for p in raw_preds]
        true_labels = df["label_str"].tolist()

        # Use only labels present in either true or pred to avoid empty CM rows
        label_set = sorted(set(true_labels) | set(pred_labels))

        return _run_validation(
            model_name="Feel-IT Emotion (Italian)",
            true_labels=true_labels,
            pred_labels=pred_labels,
            label_set=label_set,
            cm_filename="cm_feel_it_emotion.png",
        )
    except Exception as exc:
        log.error("Feel-IT emotion validation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_validation(
    max_samples: int = MAX_SAMPLES,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Run all three validations and return a summary DataFrame.

    Parameters
    ----------
    max_samples : Maximum benchmark examples per model (default 500).
    batch_size  : HuggingFace inference batch size (default 8, M2-safe).
    """
    _ensure_dirs()

    # Remove stale report so we start fresh on each full run
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
        log.info("Removed existing report at %s", REPORT_PATH)

    summaries = []
    validators = [
        validate_feel_it_sentiment,
        validate_vader_sentiment,
        validate_feel_it_emotion,
    ]
    for fn in tqdm(validators, desc="Validation suite"):
        result = fn(max_samples=max_samples, batch_size=batch_size)
        if result:
            summaries.append(result)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        print("\n" + "=" * 60)
        print("  VALIDATION SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        print(f"\nFull report saved → {REPORT_PATH}")
        return summary_df

    log.warning("No validation results produced.")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate NLP models.")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                        help=f"Max benchmark examples per model (default: {MAX_SAMPLES}).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"HuggingFace batch size (default: {BATCH_SIZE}).")
    args = parser.parse_args()

    run_validation(max_samples=args.max_samples, batch_size=args.batch_size)
