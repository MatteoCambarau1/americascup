"""
Rigenera le 3 wordcloud incluse in report_tecnico.md con uno stile più
professionale, leggendo direttamente i testi lemmatizzati da
data/processed/reviews_topics.csv (nessun bisogno di rifittare BERTopic).

Wordcloud rigenerate
---------------------
- wordcloud_cagliari_pre_positive.png
- wordcloud_cagliari_during_negative.png
- wordcloud_olbia_pre_positive.png

Uso
---
    python -m analysis.regenerate_wordclouds
"""

from __future__ import annotations

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from analysis.config import PROCESSED_DATA_DIR, RESULTS_DIR
from analysis.plot_style import PALETTE, apply_style

apply_style()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TOPICS_FILE = os.path.join(PROCESSED_DATA_DIR, "reviews_topics.csv")
OUT_DIR     = os.path.join(RESULTS_DIR, "topic_modeling")

TEXT_COLS = {"positive": "text_positive_lemma", "negative": "text_negative_lemma"}

# (city, temporal_window, sentiment) -> usati nel report
TARGETS = [
    ("cagliari", "pre",    "positive"),
    ("cagliari", "during", "negative"),
    ("olbia",    "pre",    "positive"),
]


def main() -> None:
    log.info("Loading %s ...", TOPICS_FILE)
    df = pd.read_csv(TOPICS_FILE, dtype=str, low_memory=False)
    df["city"] = df["city"].str.lower().str.strip()

    os.makedirs(OUT_DIR, exist_ok=True)

    for city, window, sentiment in TARGETS:
        mask = (df["city"] == city) & (df["temporal_window"] == window)
        subset = df[mask]
        col = TEXT_COLS[sentiment]
        texts = subset[col].dropna().tolist() if col in subset.columns else []
        valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]

        if not valid_texts:
            log.warning("[%s/%s/%s] Nessun testo valido — skip.", city, window, sentiment)
            continue

        combined = " ".join(valid_texts)
        colormap = "Greens" if sentiment == "positive" else "Reds"
        accent   = PALETTE["positive"] if sentiment == "positive" else PALETTE["negative"]

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap=colormap,
            max_words=100,
            collocations=False,
            prefer_horizontal=0.95,
            relative_scaling=0.4,
            margin=4,
        ).generate(combined)

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        label_it = {"positive": "recensioni positive", "negative": "recensioni negative"}[sentiment]
        window_it = {"pre": "pre-evento", "during": "durante l'evento", "post": "post-evento"}[window]
        ax.set_title(
            f"Word Cloud — {city.title()}, {window_it} ({label_it})\n"
            f"n = {len(valid_texts)} recensioni",
            fontsize=14, fontweight="bold", color="#333333", pad=14,
        )

        # Cornice colorata coerente con il sentiment
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(accent)
            spine.set_linewidth(2.5)

        plt.tight_layout()

        fname = f"wordcloud_{city}_{window}_{sentiment}.png"
        out_png = os.path.join(OUT_DIR, fname)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        log.info("Saved %s (n=%d)", fname, len(valid_texts))


if __name__ == "__main__":
    main()
