"""
Stile grafico condiviso per i chart del report tecnico
"America's Cup Cagliari".

Importare `apply_style()` e chiamarla una volta all'inizio di qualsiasi
script che produce grafici destinati a `report_tecnico.md`
(exploratory.py, empirical_validation.py, topic_modeling.py), per garantire
palette colori coerente, font uniforme e griglie leggere su tutte le figure.

Uso
---
    import matplotlib
    matplotlib.use("Agg")
    from analysis.plot_style import apply_style, PALETTE
    apply_style()
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Palette coerente usata in tutto il report
# ---------------------------------------------------------------------------
PALETTE = {
    # Confronto 2026 (evento) vs 2025 (baseline)
    "year_2026": "#1f6fb4",
    "year_2025": "#a9cce8",
    # Finestre temporali pre / during / post
    "pre":    "#2ca02c",
    "during": "#d62728",
    "post":   "#9467bd",
    # Sentiment
    "positive": "#2ca02c",
    "negative": "#d62728",
    # Accenti
    "accent":  "#F5C518",
    "neutral": "#94A3B8",
    # Modelli di validazione
    "feelit":  "#127EEA",
    "nlptown": "#E8650A",
}

# Etichette leggibili per le finestre temporali (con a-capo per spazio)
WINDOW_LABELS = {
    "pre":    "Pre\n(1 apr–20 mag)",
    "during": "During\n(21–24 mag)",
    "post":   "Post\n(25 mag–15 giu)",
}


def apply_style() -> None:
    """Applica lo stile seaborn 'whitegrid' + font/dimensioni coerenti."""
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "grid.color": "#e3e7eb",
        "grid.linewidth": 0.8,
        "font.size": 10,
    })


def annotate_bar(ax, bar, text: str, fontsize: int = 9, color: str = "#333333") -> None:
    """Scrive `text` centrato sopra una barra (helper per annotazioni)."""
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        text,
        ha="center", va="bottom", fontsize=fontsize, color=color,
    )
