"""
Topic modelling pipeline for tourist review data.

Two complementary approaches:
    LDA     — sklearn LatentDirichletAllocation, fast and interpretable.
    BERTopic — transformer-based clustering with paraphrase-multilingual-MiniLM-L12-v2,
               works across Italian and English without language switching.

Both models are run separately on text_positive_lemma and text_negative_lemma.

Additional analyses (per city × temporal-window subset)
--------------------------------------------------------
GridSearch  — finds the optimal (n_topics, learning_decay) via cross-validated
              log-likelihood; saves a log-likelihood curve PNG per subset.
pyLDAvis    — interactive HTML visualisation of the fitted LDA model.
WordClouds  — positive (green) and negative (red) word-cloud PNGs.

Outputs
-------
data/processed/reviews_topics.csv          — full dataframe with topic columns
results/topics_by_window.csv               — topic distribution per temporal window
results/topics_by_city.csv                 — topic distribution per city
results/lda_top_words.csv                  — top-N words per LDA topic
results/bertopic_top_words.csv             — top-N words per BERTopic topic
results/topic_modeling/loglikelihood_{city}_{window}.png
results/topic_modeling/pyldavis_{city}_{window}.html
results/topic_modeling/wordcloud_{city}_{window}_positive.png
results/topic_modeling/wordcloud_{city}_{window}_negative.png

Dependencies
------------
pip install bertopic sentence-transformers scikit-learn pyldavis wordcloud
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on M2 without a display
import matplotlib.pyplot as plt
import numpy as np
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
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

INPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "reviews_processed.csv")
TOPICS_OUTPUT = os.path.join(PROCESSED_DATA_DIR, "reviews_topics.csv")
AGG_WINDOW_OUTPUT = os.path.join(RESULTS_DIR, "topics_by_window.csv")
AGG_CITY_OUTPUT = os.path.join(RESULTS_DIR, "topics_by_city.csv")
LDA_WORDS_OUTPUT = os.path.join(RESULTS_DIR, "lda_top_words.csv")
BERTOPIC_WORDS_OUTPUT = os.path.join(RESULTS_DIR, "bertopic_top_words.csv")

# Sub-directory for per-subset visualisations
TOPIC_MODELING_DIR = os.path.join(RESULTS_DIR, "topic_modeling")

# Text columns to model
TEXT_COLS = {
    "positive": "text_positive_lemma",
    "negative": "text_negative_lemma",
}

# BERTopic sentence-transformer model — multilingual, ~120 MB, M2-friendly
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Default number of top words to extract per topic
TOP_N_WORDS = 10

# GridSearch parameter grid for LDA
GRID_N_TOPICS: list[int] = list(range(3, 9))       # [3, 4, 5, 6, 7, 8]
GRID_LEARNING_DECAY: list[float] = [0.5, 0.7, 0.9]

# Minimum documents required to attempt pyLDAvis rendering
MIN_DOCS_PYLDAVIS = 50

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_processed(path: str = INPUT_FILE) -> pd.DataFrame:
    """Load the preprocessed review CSV."""
    log.info("Loading processed reviews from '%s' …", path)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    log.info("Loaded %d rows.", len(df))
    return df


def _save(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved → %s", path)


def _slug(s: str) -> str:
    """Convert an arbitrary string into a safe filename component."""
    return re.sub(r"[^\w]+", "_", s.lower()).strip("_")


# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------

def _build_vectorizer(texts: list[str]):
    """Return a fitted CountVectorizer and the document-term matrix."""
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(
        max_df=0.95,        # ignore terms in >95 % of docs (too common)
        min_df=2,           # ignore terms that appear in fewer than 2 docs
        max_features=5_000,
    )
    dtm = vec.fit_transform(texts)
    return vec, dtm


def run_lda(
    texts: list[str],
    n_topics: int = 10,
    top_n_words: int = TOP_N_WORDS,
    random_state: int = 42,
) -> tuple[list[int], list[float], pd.DataFrame]:
    """
    Fit LDA on *texts*.

    Returns
    -------
    dominant_topics : list[int]   — most probable topic index per document
    topic_probs     : list[float] — probability of the dominant topic
    top_words_df    : DataFrame   — columns [topic, word, rank]
    """
    from sklearn.decomposition import LatentDirichletAllocation

    # Filter empty documents; keep track of original indices
    valid_idx = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    valid_texts = [texts[i] for i in valid_idx]

    if not valid_texts:
        log.warning("No valid texts for LDA.")
        return (
            [-1] * len(texts),
            [0.0] * len(texts),
            pd.DataFrame(columns=["topic", "word", "rank"]),
        )

    log.info("Fitting LDA with %d topics on %d documents …", n_topics, len(valid_texts))
    vec, dtm = _build_vectorizer(valid_texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        n_jobs=-1,          # use all CPU cores
    )

    log.info("LDA fitting …")
    doc_topic_matrix = lda.fit_transform(dtm)

    # Dominant topic and its probability per valid document
    dominant = doc_topic_matrix.argmax(axis=1).tolist()
    probs = doc_topic_matrix.max(axis=1).tolist()

    # Map back to original indices (invalid docs get -1 / 0.0)
    full_dominant = [-1] * len(texts)
    full_probs = [0.0] * len(texts)
    for pos, orig_i in enumerate(valid_idx):
        full_dominant[orig_i] = dominant[pos]
        full_probs[orig_i] = round(probs[pos], 4)

    # Top-N words per topic
    feature_names = vec.get_feature_names_out()
    rows = []
    for topic_idx, topic_vec in enumerate(lda.components_):
        top_indices = topic_vec.argsort()[: -top_n_words - 1 : -1]
        for rank, word_idx in enumerate(top_indices):
            rows.append({
                "topic": topic_idx,
                "word": feature_names[word_idx],
                "rank": rank + 1,
            })
    top_words_df = pd.DataFrame(rows)

    return full_dominant, full_probs, top_words_df


# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------

def _build_bertopic_model(n_topics: Optional[int]) -> object:
    """Instantiate a BERTopic model with a lightweight multilingual embedder."""
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    log.info("Loading sentence-transformer: %s …", SBERT_MODEL)
    embedder = SentenceTransformer(SBERT_MODEL)

    nr_topics = n_topics if n_topics is not None else "auto"
    model = BERTopic(
        embedding_model=embedder,
        nr_topics=nr_topics,
        verbose=False,
        calculate_probabilities=True,
    )
    return model


def _encode_in_batches(
    model,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Encode texts in batches to avoid OOM on 8 GB RAM."""
    embedder = model.embedding_model
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding embeddings"):
        batch = texts[i : i + batch_size]
        embeddings.append(embedder.encode(batch, show_progress_bar=False))
    return np.vstack(embeddings)


def run_bertopic(
    texts: list[str],
    n_topics: Optional[int] = None,
    top_n_words: int = TOP_N_WORDS,
    batch_size: int = 8,
) -> tuple[list[int], list[float], pd.DataFrame]:
    """
    Fit BERTopic on *texts*.

    Returns
    -------
    topics      : list[int]   — topic index per document (-1 = outlier)
    probs       : list[float] — probability of the assigned topic
    top_words_df: DataFrame   — columns [topic, word, rank]
    """
    valid_idx = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    valid_texts = [texts[i] for i in valid_idx]

    if len(valid_texts) < 10:
        log.warning("Too few valid texts for BERTopic (%d). Skipping.", len(valid_texts))
        return (
            [-1] * len(texts),
            [0.0] * len(texts),
            pd.DataFrame(columns=["topic", "word", "rank"]),
        )

    log.info("Running BERTopic on %d documents …", len(valid_texts))
    model = _build_bertopic_model(n_topics)

    # Embed in batches, then pass pre-computed embeddings to fit_transform
    embeddings = _encode_in_batches(model, valid_texts, batch_size)

    log.info("Fitting BERTopic …")
    doc_topics, doc_probs = model.fit_transform(valid_texts, embeddings=embeddings)

    # doc_probs can be None when calculate_probabilities=True but corpus is tiny
    if doc_probs is None:
        doc_probs = [0.0] * len(valid_texts)
    else:
        # Shape may be (n_docs,) or (n_docs, n_topics)
        if hasattr(doc_probs, "ndim") and doc_probs.ndim == 2:
            doc_probs = doc_probs.max(axis=1).tolist()
        else:
            doc_probs = list(doc_probs)

    # Map back to original indices
    full_topics = [-1] * len(texts)
    full_probs = [0.0] * len(texts)
    for pos, orig_i in enumerate(valid_idx):
        full_topics[orig_i] = int(doc_topics[pos])
        full_probs[orig_i] = round(float(doc_probs[pos]), 4)

    # Top-N words per topic
    topic_info = model.get_topics()
    rows = []
    for topic_id, word_score_pairs in topic_info.items():
        if topic_id == -1:  # outlier cluster
            continue
        for rank, (word, _score) in enumerate(word_score_pairs[:top_n_words]):
            rows.append({"topic": topic_id, "word": word, "rank": rank + 1})
    top_words_df = pd.DataFrame(rows)

    return full_topics, full_probs, top_words_df


# ---------------------------------------------------------------------------
# GridSearch — optimal number of LDA topics
# ---------------------------------------------------------------------------

def find_optimal_n_topics(
    texts: list[str],
    label: str,
    n_topics_range: list[int] = GRID_N_TOPICS,
    learning_decay_values: list[float] = GRID_LEARNING_DECAY,
    random_state: int = 42,
) -> int:
    """
    Use GridSearchCV to find the (n_topics, learning_decay) combination that
    maximises cross-validated log-likelihood on *texts*.

    Saves a log-likelihood curve PNG (one line per learning_decay value) to
    results/topic_modeling/loglikelihood_{label}.png.

    Returns the optimal n_topics as an integer.
    Falls back to the midpoint of *n_topics_range* on any error.
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.model_selection import GridSearchCV

    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    fallback_n = n_topics_range[len(n_topics_range) // 2]

    if len(valid_texts) < 20:
        log.warning(
            "[%s] Too few documents for GridSearch (%d) — using n_topics=%d.",
            label, len(valid_texts), fallback_n,
        )
        return fallback_n

    log.info("[%s] GridSearch over n_topics=%s, learning_decay=%s …",
             label, n_topics_range, learning_decay_values)

    try:
        vec, dtm = _build_vectorizer(valid_texts)

        param_grid = {
            "n_components": n_topics_range,
            "learning_decay": learning_decay_values,
        }
        lda_base = LatentDirichletAllocation(
            random_state=random_state,
            n_jobs=-1,
            max_iter=15,        # keep fast for grid search; final fit uses more
        )
        search = GridSearchCV(
            lda_base,
            param_grid,
            cv=3,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        search.fit(dtm)

        best_n = search.best_params_["n_components"]
        best_decay = search.best_params_["learning_decay"]
        best_ll = round(search.best_score_, 2)
        log.info(
            "[%s] Best: n_topics=%d, learning_decay=%.1f  (log-likelihood=%.2f)",
            label, best_n, best_decay, best_ll,
        )
        print(
            f"  [{label}] Optimal n_topics = {best_n}  "
            f"(decay={best_decay}, log-likelihood={best_ll})"
        )

        # ── Log-likelihood curve ──────────────────────────────────────────
        results_df = pd.DataFrame(search.cv_results_)
        Path(TOPIC_MODELING_DIR).mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        for decay in learning_decay_values:
            mask = results_df["param_learning_decay"] == decay
            sub = results_df[mask].sort_values("param_n_components")
            ax.plot(
                sub["param_n_components"],
                sub["mean_test_score"],
                marker="o",
                label=f"decay={decay}",
            )

        ax.set_xlabel("Number of Topics")
        ax.set_ylabel("Mean Log-Likelihood (CV)")
        ax.set_title(f"LDA Grid Search — {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_png = os.path.join(TOPIC_MODELING_DIR, f"loglikelihood_{_slug(label)}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        log.info("Log-likelihood plot saved → %s", out_png)

        return best_n

    except Exception as exc:
        log.warning("[%s] GridSearch failed (%s) — using n_topics=%d.", label, exc, fallback_n)
        return fallback_n


# ---------------------------------------------------------------------------
# pyLDAvis — interactive HTML visualisation
# ---------------------------------------------------------------------------

def generate_pyldavis(
    lda_model,
    vectorizer,
    dtm,
    label: str,
) -> None:
    """
    Prepare a pyLDAvis visualisation for *lda_model* and save it as an HTML
    file to results/topic_modeling/pyldavis_{label}.html.

    Silently skips if pyLDAvis is not installed or the model is too small.
    """
    try:
        # pyLDAvis ≥ 3.3 moved the sklearn helper to pyLDAvis.lda_model
        try:
            import pyLDAvis.lda_model as pyldavis_sklearn
        except ImportError:
            import pyLDAvis.sklearn as pyldavis_sklearn  # older versions
        import pyLDAvis

        pyLDAvis.enable_notebook()  # suppresses interactive display in scripts

        log.info("[%s] Preparing pyLDAvis …", label)
        vis = pyldavis_sklearn.prepare(lda_model, dtm, vectorizer, mds="tsne")

        Path(TOPIC_MODELING_DIR).mkdir(parents=True, exist_ok=True)
        out_html = os.path.join(TOPIC_MODELING_DIR, f"pyldavis_{_slug(label)}.html")
        pyLDAvis.save_html(vis, out_html)
        log.info("pyLDAvis saved → %s", out_html)

    except ImportError:
        log.warning("[%s] pyLDAvis not installed — skipping HTML visualisation.", label)
    except Exception as exc:
        log.warning("[%s] pyLDAvis failed (%s) — skipping.", label, exc)


# ---------------------------------------------------------------------------
# WordClouds
# ---------------------------------------------------------------------------

def generate_wordcloud(
    texts: list[str],
    label: str,
    sentiment: str,
) -> None:
    """
    Generate a word cloud from *texts* and save it as a PNG.

    Parameters
    ----------
    texts     : Pre-processed / lemmatized text strings.
    label     : Used in the output filename (e.g. "cagliari_pre").
    sentiment : "positive" → green palette; "negative" → red palette.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        log.warning("wordcloud not installed — skipping WordCloud generation.")
        return

    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not valid_texts:
        log.warning("[%s/%s] No valid texts for WordCloud — skipping.", label, sentiment)
        return

    combined = " ".join(valid_texts)
    colormap = "Greens" if sentiment == "positive" else "Reds"

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=colormap,
        max_words=100,
        collocations=False,
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud — {label} ({sentiment})", fontsize=13)
    plt.tight_layout()

    Path(TOPIC_MODELING_DIR).mkdir(parents=True, exist_ok=True)
    out_png = os.path.join(
        TOPIC_MODELING_DIR,
        f"wordcloud_{_slug(label)}_{sentiment}.png",
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info("WordCloud saved → %s", out_png)


# ---------------------------------------------------------------------------
# Per-subset LDA helper (used for pyLDAvis — separate from the main fit)
# ---------------------------------------------------------------------------

def _fit_lda_for_viz(texts: list[str], n_topics: int, random_state: int = 42):
    """
    Fit a fresh LDA on *texts* and return (vectorizer, dtm, lda_model).
    Returns None on failure (e.g. too few documents).
    """
    from sklearn.decomposition import LatentDirichletAllocation

    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(valid_texts) < MIN_DOCS_PYLDAVIS:
        return None

    try:
        vec, dtm = _build_vectorizer(valid_texts)
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            n_jobs=-1,
        )
        lda.fit(dtm)
        return vec, dtm, lda
    except Exception as exc:
        log.warning("_fit_lda_for_viz failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def _distribution(
    df: pd.DataFrame,
    group_col: str,
    topic_col: str,
) -> pd.DataFrame:
    """
    Compute topic count distribution grouped by *group_col*.
    Returns a wide dataframe with one column per topic value.
    """
    if group_col not in df.columns or topic_col not in df.columns:
        return pd.DataFrame()

    dist = (
        df.groupby([group_col, topic_col])
        .size()
        .reset_index(name="count")
        .pivot(index=group_col, columns=topic_col, values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    # Prefix columns so multiple topic columns can be merged
    dist.columns = (
        [group_col]
        + [f"{topic_col}_{c}" for c in dist.columns if c != group_col]
    )
    return dist


def aggregate_by_window(df: pd.DataFrame) -> pd.DataFrame:
    """Topic distributions for both LDA and BERTopic, grouped by temporal_window."""
    frames = [df[["temporal_window"]].drop_duplicates()] if "temporal_window" in df.columns else []
    for prefix in ("positive", "negative"):
        for model in ("lda", "bertopic"):
            col = f"{model}_topic_{prefix}"
            if col in df.columns:
                dist = _distribution(df, "temporal_window", col)
                if not dist.empty:
                    frames.append(dist.set_index("temporal_window"))

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames[1:], axis=1).reset_index()
    result.columns.name = None
    return result


def aggregate_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """Topic distributions for both LDA and BERTopic, grouped by city."""
    city_col = "property_city" if "property_city" in df.columns else ("city" if "city" in df.columns else None)
    if city_col is None:
        return pd.DataFrame()
    frames = [df[[city_col]].drop_duplicates()]
    for prefix in ("positive", "negative"):
        for model in ("lda", "bertopic"):
            col = f"{model}_topic_{prefix}"
            if col in df.columns:
                dist = _distribution(df, city_col, col)
                if not dist.empty:
                    frames.append(dist.set_index(city_col))

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames[1:], axis=1).reset_index()
    result.columns.name = None
    return result


# ---------------------------------------------------------------------------
# Per-subset visualisations (GridSearch + pyLDAvis + WordClouds)
# ---------------------------------------------------------------------------

def run_subset_visualisations(df: pd.DataFrame) -> None:
    """
    For each (city, temporal_window) combination:
      1. GridSearch → optimal n_topics → log-likelihood PNG
      2. Fit LDA on positive texts with optimal n_topics → pyLDAvis HTML
      3. WordCloud PNG for positive texts (green)
      4. WordCloud PNG for negative texts (red)

    Skips any subset with fewer than MIN_DOCS_PYLDAVIS valid documents.
    All errors are caught per-subset so the pipeline never aborts.
    """
    city_col = "property_city" if "property_city" in df.columns else ("city" if "city" in df.columns else None)
    window_col = "temporal_window"
    pos_col = TEXT_COLS["positive"]
    neg_col = TEXT_COLS["negative"]

    if city_col is None or window_col not in df.columns:
        log.warning("City or window column missing — skipping subset visualisations.")
        return

    cities = df[city_col].dropna().unique().tolist()
    windows = df[window_col].dropna().unique().tolist()
    subsets = [(c, w) for c in cities for w in windows]

    for city, window in tqdm(subsets, desc="Subset visualisations"):
        mask = (df[city_col] == city) & (df[window_col] == window)
        subset = df[mask]
        if subset.empty:
            continue

        label = f"{city}_{window}"
        pos_texts = subset[pos_col].fillna("").tolist() if pos_col in subset.columns else []
        neg_texts = subset[neg_col].fillna("").tolist() if neg_col in subset.columns else []

        valid_pos = [t for t in pos_texts if t.strip()]
        log.info("[%s] %d positive docs, %d negative docs.", label, len(valid_pos),
                 len([t for t in neg_texts if t.strip()]))

        # 1 ── GridSearch on positive texts
        optimal_n = find_optimal_n_topics(pos_texts, label=label)

        # 2 ── pyLDAvis (positive texts only; needs ≥ MIN_DOCS_PYLDAVIS docs)
        if len(valid_pos) >= MIN_DOCS_PYLDAVIS:
            viz_result = _fit_lda_for_viz(pos_texts, n_topics=optimal_n)
            if viz_result is not None:
                vec, dtm, lda_model = viz_result
                generate_pyldavis(lda_model, vec, dtm, label=label)
        else:
            log.info(
                "[%s] Only %d docs — pyLDAvis skipped (min=%d).",
                label, len(valid_pos), MIN_DOCS_PYLDAVIS,
            )

        # 3 ── WordClouds
        generate_wordcloud(pos_texts, label=label, sentiment="positive")
        generate_wordcloud(neg_texts, label=label, sentiment="negative")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str = INPUT_FILE,
    n_topics: int = 10,
    batch_size: int = 8,
    top_n_words: int = TOP_N_WORDS,
    sample_size: Optional[int] = None,
    run_gridsearch: bool = True,
) -> pd.DataFrame:
    """
    Execute the full topic modelling pipeline.

    Parameters
    ----------
    input_path     : Path to the preprocessed CSV (output of preprocessing.py).
    n_topics       : Fallback number of topics used when GridSearch is disabled.
    batch_size     : Embedding batch size for BERTopic (tuned for M2 8 GB).
    top_n_words    : Number of representative words to extract per topic.
    sample_size    : If set, run only on the first N rows (useful for quick tests).
    run_gridsearch : If True, find optimal n_topics via GridSearch before fitting
                     LDA; if False, use the *n_topics* parameter directly.
    """
    df = load_processed(input_path)
    if df.empty:
        log.warning("Input dataframe is empty — aborting.")
        return df

    if sample_size is not None:
        log.info("Sample mode: using first %d rows.", sample_size)
        df = df.head(sample_size).copy()

    all_lda_words: list[pd.DataFrame] = []
    all_bertopic_words: list[pd.DataFrame] = []

    for polarity, col in TEXT_COLS.items():
        # Fall back to raw text column if lemmatized version is absent
        if col not in df.columns:
            raw_col = col.replace("_lemma", "")
            log.warning("'%s' not found — falling back to '%s'.", col, raw_col)
            df[col] = df.get(raw_col, pd.Series(dtype=str))

        texts = df[col].fillna("").tolist()

        # Determine n_topics: GridSearch on full corpus or use fixed value
        if run_gridsearch:
            effective_n_topics = find_optimal_n_topics(texts, label=f"full_{polarity}")
        else:
            effective_n_topics = n_topics

        # --- LDA ---
        log.info("=== LDA — %s (n_topics=%d) ===", polarity, effective_n_topics)
        lda_topics, lda_probs, lda_words = run_lda(
            texts, n_topics=effective_n_topics, top_n_words=top_n_words
        )
        df[f"lda_topic_{polarity}"] = lda_topics
        df[f"lda_topic_{polarity}_prob"] = lda_probs
        if not lda_words.empty:
            lda_words["polarity"] = polarity
            all_lda_words.append(lda_words)

        # --- BERTopic ---
        log.info("=== BERTopic — %s ===", polarity)
        bert_topics, bert_probs, bert_words = run_bertopic(
            texts, n_topics=effective_n_topics, top_n_words=top_n_words,
            batch_size=batch_size,
        )
        df[f"bertopic_topic_{polarity}"] = bert_topics
        df[f"bertopic_topic_{polarity}_prob"] = bert_probs
        if not bert_words.empty:
            bert_words["polarity"] = polarity
            all_bertopic_words.append(bert_words)

    # Per-subset visualisations (GridSearch plots, pyLDAvis, WordClouds)
    log.info("=== Subset visualisations ===")
    run_subset_visualisations(df)

    # Aggregations
    agg_window = aggregate_by_window(df)
    agg_city = aggregate_by_city(df)

    # Save outputs
    _save(df, TOPICS_OUTPUT)
    if agg_window is not None and not agg_window.empty:
        _save(agg_window, AGG_WINDOW_OUTPUT)
    if agg_city is not None and not agg_city.empty:
        _save(agg_city, AGG_CITY_OUTPUT)
    if all_lda_words:
        _save(pd.concat(all_lda_words, ignore_index=True), LDA_WORDS_OUTPUT)
    if all_bertopic_words:
        _save(pd.concat(all_bertopic_words, ignore_index=True), BERTOPIC_WORDS_OUTPUT)

    log.info("Pipeline complete. Shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run topic modelling pipeline.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only the first N rows (for testing).")
    parser.add_argument("--n-topics", type=int, default=10,
                        help="Fallback number of topics when --no-gridsearch is set (default: 10).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Embedding batch size for BERTopic (default: 8).")
    parser.add_argument("--top-n-words", type=int, default=TOP_N_WORDS,
                        help=f"Top words per topic (default: {TOP_N_WORDS}).")
    parser.add_argument("--no-gridsearch", action="store_true",
                        help="Skip GridSearch and use --n-topics directly.")
    args = parser.parse_args()

    run_pipeline(
        n_topics=args.n_topics,
        batch_size=args.batch_size,
        top_n_words=args.top_n_words,
        sample_size=args.sample,
        run_gridsearch=not args.no_gridsearch,
    )
