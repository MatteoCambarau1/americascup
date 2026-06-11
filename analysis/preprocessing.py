"""
Preprocessing pipeline for tourist review data.

Steps:
    1. Load and combine all CSVs from data/raw/
    2. Normalize placeholder values ("---") in text_negative to NaN
    3. Parse Italian-format dates (review_date, stay_date)
    4. Assign temporal window to each review
    5. Clean review text (lowercase, remove punctuation / special chars)
    6. Remove multilingual stopwords (Italian + English) via NLTK
    7. Lemmatize tokens with SpaCy (it_core_news_sm / en_core_web_sm)
    8. Feature engineering: review_length, has_negative, temporal_window
    9. Save processed dataframe to data/processed/reviews_processed.csv
"""

from __future__ import annotations

import os
import re
import glob
import logging
from pathlib import Path
from typing import Optional

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords

from analysis.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TARGET_CITY,
    CONTROL_CITY,
)
from analysis.utils import parse_italian_date, assign_temporal_window

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

PLACEHOLDER = "---"
OUTPUT_FILENAME = "reviews_processed.csv"

# Columns that contain review text to be cleaned / lemmatized
TEXT_COLS = ["text_positive", "text_negative", "review_title"]

# Date columns to parse
DATE_COLS = ["review_date", "stay_date"]

# ---------------------------------------------------------------------------
# NLP model loading (done once at module level)
# ---------------------------------------------------------------------------

def _load_stopwords() -> set[str]:
    """Download NLTK stopword lists if needed and return a combined set."""
    for resource in ("stopwords", "punkt"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            log.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)

    it_sw = set(stopwords.words("italian"))
    en_sw = set(stopwords.words("english"))
    return it_sw | en_sw


def _load_spacy_model(model_name: str) -> spacy.language.Language:
    """Load a SpaCy model, raising a clear error if it is not installed."""
    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        raise OSError(
            f"SpaCy model '{model_name}' not found. "
            f"Install it with:\n  python -m spacy download {model_name}"
        )


log.info("Loading NLP resources …")
STOPWORDS: set[str] = _load_stopwords()
NLP_IT: spacy.language.Language = _load_spacy_model("it_core_news_sm")
NLP_EN: spacy.language.Language = _load_spacy_model("en_core_web_sm")
log.info("NLP resources ready.")

# ---------------------------------------------------------------------------
# Step 1 — Load CSVs
# ---------------------------------------------------------------------------

def load_raw_reviews(raw_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Read every CSV file inside *raw_dir* and concatenate them into one
    dataframe.  Returns an empty dataframe with the expected schema if no
    files are found.
    """
    pattern = os.path.join(raw_dir, "*.csv")
    files = glob.glob(pattern)

    if not files:
        log.warning("No CSV files found in '%s'.", raw_dir)
        return pd.DataFrame()

    frames = []
    for path in files:
        log.info("Reading %s", path)
        df = pd.read_csv(path, dtype=str)  # read as str to preserve raw dates
        df["scrape_source"] = df.get("scrape_source", pd.Series(dtype=str))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d reviews from %d file(s).", len(combined), len(files))
    return combined


# ---------------------------------------------------------------------------
# Step 2 — Normalize placeholders
# ---------------------------------------------------------------------------

def normalize_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '---' placeholder in text_negative with NaN."""
    if "text_negative" in df.columns:
        df["text_negative"] = df["text_negative"].replace(PLACEHOLDER, pd.NA)
    return df


# ---------------------------------------------------------------------------
# Step 3 — Parse dates
# ---------------------------------------------------------------------------

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse review_date and stay_date from Italian-format strings using
    parse_italian_date().  Unparseable values become NaT.
    """
    for col in DATE_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(
            lambda v: parse_italian_date(v) if pd.notna(v) else pd.NaT
        )
    return df


# ---------------------------------------------------------------------------
# Step 4 — Assign temporal window
# ---------------------------------------------------------------------------

def assign_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'temporal_window' column by calling assign_temporal_window() on
    review_date.  Rows with missing review_date receive 'unknown'.
    """
    def _window(row: pd.Series) -> str:
        rd = row.get("review_date")
        if pd.isna(rd):
            return "unknown"
        return assign_temporal_window(rd, rd.year)

    df["temporal_window"] = df.apply(_window, axis=1)
    return df


# ---------------------------------------------------------------------------
# Step 5 — Text cleaning
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    """
    Lowercase, remove punctuation and special characters, collapse whitespace.
    Returns an empty string for null / non-string input.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text() to all TEXT_COLS that are present in the dataframe."""
    for col in TEXT_COLS:
        if col in df.columns:
            cleaned_col = f"{col}_clean"
            df[cleaned_col] = df[col].apply(clean_text)
    return df


# ---------------------------------------------------------------------------
# Step 6 — Stopword removal
# ---------------------------------------------------------------------------

def remove_stopwords(text: str) -> str:
    """Remove Italian and English stopwords from a pre-cleaned string."""
    tokens = text.split()
    filtered = [t for t in tokens if t not in STOPWORDS]
    return " ".join(filtered)


def remove_stopwords_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply remove_stopwords() to every *_clean column."""
    clean_cols = [c for c in df.columns if c.endswith("_clean")]
    for col in clean_cols:
        df[col] = df[col].apply(remove_stopwords)
    return df


# ---------------------------------------------------------------------------
# Step 7 — Lemmatization
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> str:
    """
    Heuristic language detection: returns 'it' or 'en'.
    Uses the proportion of Italian stopwords present in the token list.
    Defaults to 'it' when evidence is insufficient (project is Italy-centric).
    """
    if not text:
        return "it"
    tokens = set(text.split())
    it_hits = len(tokens & set(stopwords.words("italian")))
    en_hits = len(tokens & set(stopwords.words("english")))
    return "en" if en_hits > it_hits else "it"


def lemmatize_text(text: str) -> str:
    """
    Lemmatize *text* using the SpaCy model appropriate for its detected
    language.  Returns a space-joined string of lemmas.
    """
    if not text:
        return ""
    lang = _detect_language(text)
    nlp = NLP_EN if lang == "en" else NLP_IT
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if token.lemma_.strip())


def lemmatize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lemmatize_text() to every *_clean column, producing *_lemma columns."""
    clean_cols = [c for c in df.columns if c.endswith("_clean")]
    for col in clean_cols:
        base = col.replace("_clean", "")
        lemma_col = f"{base}_lemma"
        log.info("Lemmatizing column: %s → %s", col, lemma_col)
        df[lemma_col] = df[col].apply(lemmatize_text)
    return df


# ---------------------------------------------------------------------------
# Step 8 — Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns:
        review_length  — word count of text_positive
        has_negative   — True when text_negative is not NaN
    (temporal_window is already added by assign_windows)
    """
    if "text_positive" in df.columns:
        df["review_length"] = df["text_positive"].apply(
            lambda t: len(str(t).split()) if pd.notna(t) else 0
        )

    if "text_negative" in df.columns:
        df["has_negative"] = df["text_negative"].notna()

    return df


# ---------------------------------------------------------------------------
# Step 9 — Save
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, output_dir: str = PROCESSED_DATA_DIR) -> str:
    """
    Save *df* to <output_dir>/reviews_processed.csv.
    Creates the output directory if it does not exist.
    Returns the full output path.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, OUTPUT_FILENAME)
    df.to_csv(out_path, index=False)
    log.info("Saved %d rows to '%s'.", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_dir: str = RAW_DATA_DIR,
    output_dir: str = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline and return the processed dataframe.
    """
    df = load_raw_reviews(raw_dir)
    if df.empty:
        log.warning("Empty dataframe — pipeline aborted.")
        return df

    df = normalize_placeholders(df)
    df = parse_dates(df)
    df = assign_windows(df)
    df = clean_text_columns(df)
    df = remove_stopwords_columns(df)
    df = lemmatize_columns(df)
    df = add_features(df)

    save_processed(df, output_dir)
    log.info("Pipeline complete. Shape: %s", df.shape)
    return df


if __name__ == "__main__":
    run_pipeline()
