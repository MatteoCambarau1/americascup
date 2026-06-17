"""
NLP model validation module.

Validates the three models used in the pipeline against public benchmark
datasets before running inference on the full review corpus.

Models validated
----------------
1. nlptown sentiment — Italian
   (nlptown/bert-base-multilingual-uncased-sentiment)
   Benchmark: cardiffnlp/tweet_sentiment_multilingual — Italian split

2. nlptown sentiment — English
   (nlptown/bert-base-multilingual-uncased-sentiment)
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
MODEL_NLPTOWN = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_FEEL_IT_EMOTION = "MilaNLProc/feel-it-italian-emotion"

# HuggingFace dataset IDs
DS_TWEET_MULTILINGUAL = "mteb/tweet_sentiment_multilingual"
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
    Load the MTEB tweet-sentiment dataset for *lang* ('italian' or 'english').

    Label mapping (dataset integers → strings):
        0 → negative, 1 → neutral, 2 → positive
    """
    from datasets import load_dataset

    log.info("Loading %s — language: %s …", DS_TWEET_MULTILINGUAL, lang)
    ds = load_dataset(DS_TWEET_MULTILINGUAL, lang, split="test")

    df = ds.to_pandas()[["text", "label"]].dropna()
    label_map = {"0": "negative", "1": "neutral", "2": "positive"}
    df["label_str"] = df["label"].map(label_map)
    df = df.dropna(subset=["label_str"])

    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=RANDOM_STATE)

    log.info("  %d examples loaded.", len(df))
    return df.reset_index(drop=True)


def _load_go_emotions_italian(max_samples: int = MAX_SAMPLES) -> pd.DataFrame:
    """
    Load dair-ai/emotion (English, public) as the Feel-IT emotion benchmark.

    NOTE: texts are English while Feel-IT was trained on Italian — expect low
    scores. Used only because MilaNLProc/de-it-fr-multilingual-go-emotions
    is access-restricted (HTTP 401).

    Label mapping (dataset integers → strings):
        0 → sadness, 1 → joy, 2 → love, 3 → anger, 4 → fear, 5 → surprise

    We keep only the four emotions in the Feel-IT label set: {joy, sadness,
    anger, fear}.
    """
    from datasets import load_dataset

    log.info("Loading dair-ai/emotion (English proxy for Feel-IT emotion benchmark) …")
    ds = load_dataset("dair-ai/emotion", split="test")

    df = ds.to_pandas()[["text", "label"]].dropna()

    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    df["label_str"] = df["label"].map(label_map)

    # Keep only emotions the Feel-IT model was trained on
    feel_it_emotions = {"joy", "sadness", "anger", "fear"}
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

def _normalise_nlptown(label: str) -> str:
    """Map nlptown star-rating output to the three-class schema.

    nlptown returns "1 star", "2 stars", … "5 stars".
    Mapping: 1-2 → negative, 3 → neutral, 4-5 → positive.
    """
    label = label.lower().strip()
    if label.startswith("1") or label.startswith("2"):
        return "negative"
    if label.startswith("3"):
        return "neutral"
    return "positive"


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
# 1. Validate nlptown sentiment (Italian)
# ---------------------------------------------------------------------------

def validate_nlptown_italian(
    max_samples: int = MAX_SAMPLES,
    batch_size: int = BATCH_SIZE,
) -> Optional[dict]:
    """Validate nlptown sentiment against the Cardiff Italian tweet dataset."""
    log.info("─── Validating: nlptown Sentiment (Italian) ───")
    try:
        df = _load_tweet_sentiment("italian", max_samples)
        pipe = _load_hf_pipeline(MODEL_NLPTOWN)

        raw_preds = _hf_predict_batched(df["text"].tolist(), pipe, batch_size)
        pred_labels = [_normalise_nlptown(p) for p in raw_preds]
        true_labels = df["label_str"].tolist()

        return _run_validation(
            model_name="nlptown Sentiment (Italian)",
            true_labels=true_labels,
            pred_labels=pred_labels,
            label_set=["positive", "negative", "neutral"],
            cm_filename="cm_nlptown_sentiment_it.png",
        )
    except Exception as exc:
        log.error("nlptown Italian sentiment validation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 2. Validate nlptown sentiment (English)
# ---------------------------------------------------------------------------

def validate_nlptown_english(
    max_samples: int = MAX_SAMPLES,
    batch_size: int = BATCH_SIZE,
) -> Optional[dict]:
    """Validate nlptown sentiment against the Cardiff English tweet dataset."""
    log.info("─── Validating: nlptown Sentiment (English) ───")
    try:
        df = _load_tweet_sentiment("english", max_samples)
        pipe = _load_hf_pipeline(MODEL_NLPTOWN)

        raw_preds = _hf_predict_batched(df["text"].tolist(), pipe, batch_size)
        pred_labels = [_normalise_nlptown(p) for p in raw_preds]
        true_labels = df["label_str"].tolist()

        return _run_validation(
            model_name="nlptown Sentiment (English)",
            true_labels=true_labels,
            pred_labels=pred_labels,
            label_set=["positive", "negative", "neutral"],
            cm_filename="cm_nlptown_sentiment_en.png",
        )
    except Exception as exc:
        log.error("nlptown English sentiment validation failed: %s", exc)
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
        validate_nlptown_italian,
        validate_nlptown_english,
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
