"""
Generazione del gold standard per la validazione empirica di Feel-IT.

Campiona 150 recensioni in italiano dai CSV grezzi di Cagliari
(30 per finestra temporale: pre/during/post 2026 + baseline_pre/baseline_during 2025),
aggiunge una colonna vuota per l'annotazione manuale e salva il file in
validation/gold_standard_da_annotare.csv pronto per Excel.

Utilizzo
--------
    python -m analysis.generate_gold_standard
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

# Rende langdetect deterministico
DetectorFactory.seed = 42

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
# Costanti
# ---------------------------------------------------------------------------

# Cartella dei CSV grezzi (relativa alla root del progetto)
RAW_DIR = Path("data/raw")

# Cartella di output
VALIDATION_DIR = Path("validation")

# File di output
OUTPUT_FILE = VALIDATION_DIR / "gold_standard_da_annotare.csv"

# Quante recensioni campionare per finestra temporale
CAMPIONE_PER_FINESTRA = 30

# Seed per riproducibilità
RANDOM_STATE = 42

# Mappa: nome file CSV → etichetta finestra temporale
# Escludiamo baseline_post (la finestra post 2025 non è inclusa per scelta progettuale)
FILE_WINDOW_MAP: dict[str, str] = {
    "booking_cagliari_2026-04-01_to_2026-05-20.csv":    "pre",
    "booking_cagliari_2026-05-21_to_2026-05-24.csv":    "during",
    "booking_cagliari_2026-05-25_to_2026-06-15.csv":    "post",
    "booking_cagliari_2025-04-01_to_2025-05-20-3.csv":  "baseline_pre",
    "booking_cagliari_2025-05-21_to_2025-05-24.csv":    "baseline_during",
}

# Colonne da tenere nel file annotato
COLONNE_OUTPUT = [
    "id", "city", "temporal_window", "review_date",
    "rating", "text_positive", "text_negative",
    "annotazione_manuale",
]

# ---------------------------------------------------------------------------
# Funzioni di utilità
# ---------------------------------------------------------------------------

def _is_italian(text: str) -> bool:
    """Restituisce True se il testo è classificato come italiano da langdetect."""
    try:
        if not isinstance(text, str) or len(text.strip()) < 10:
            return False
        return detect(text.strip()) == "it"
    except LangDetectException:
        return False


def _carica_csv_con_finestra(file_path: Path, window_label: str) -> pd.DataFrame:
    """
    Carica un CSV grezzo, assegna la finestra temporale e normalizza le colonne
    al formato atteso (il scraper usa 'city' minuscolo, 'property' senza _name).
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Assegna la finestra temporale
    df["temporal_window"] = window_label

    # Normalizzazione minima delle colonne del scraper
    # (city è già presente; review_date, rating, text_positive, text_negative idem)
    if "city" not in df.columns:
        log.warning("Colonna 'city' assente in %s — impostata a 'cagliari'.", file_path.name)
        df["city"] = "cagliari"

    # Assicura che text_negative esista anche se vuoto
    if "text_negative" not in df.columns:
        df["text_negative"] = ""

    return df


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def genera_gold_standard() -> None:
    """Campiona le recensioni, le prepara per l'annotazione e le salva."""

    os.makedirs(VALIDATION_DIR, exist_ok=True)

    campioni_per_finestra: list[pd.DataFrame] = []
    statistiche: list[dict] = []

    # ── Iterazione su ogni file / finestra temporale ─────────────────────
    for nome_file, window in FILE_WINDOW_MAP.items():
        file_path = RAW_DIR / nome_file

        if not file_path.exists():
            log.warning("File non trovato: %s — finestra '%s' saltata.", file_path, window)
            continue

        df_raw = _carica_csv_con_finestra(file_path, window)

        # Rilevamento lingua: filtriamo su text_positive non vuoto e in italiano
        df_raw = df_raw[df_raw["text_positive"].notna()]
        df_raw = df_raw[df_raw["text_positive"].str.strip() != ""]

        mascera_it = df_raw["text_positive"].apply(_is_italian)
        df_it = df_raw[mascera_it].copy()

        n_totale   = len(df_raw)
        n_italiano = len(df_it)
        n_campione = min(CAMPIONE_PER_FINESTRA, n_italiano)

        log.info(
            "Finestra %-20s | totale: %4d | italiano: %4d | campionato: %d",
            window, n_totale, n_italiano, n_campione,
        )

        statistiche.append({
            "finestra":    window,
            "totale_raw":  n_totale,
            "n_italiano":  n_italiano,
            "n_campionato": n_campione,
        })

        if n_campione == 0:
            log.warning("Nessuna recensione italiana disponibile per '%s'.", window)
            continue

        campione = df_it.sample(n=n_campione, random_state=RANDOM_STATE)
        campioni_per_finestra.append(campione)

    if not campioni_per_finestra:
        log.error("Nessun campione estratto. Controlla i file in %s.", RAW_DIR)
        return

    # ── Concatenazione e preparazione del dataset finale ─────────────────
    df_gold = pd.concat(campioni_per_finestra, ignore_index=True)

    # ID progressivo
    df_gold = df_gold.reset_index(drop=True)
    df_gold.insert(0, "id", df_gold.index + 1)

    # Colonna per l'annotazione manuale (valori attesi: positive / negative)
    df_gold["annotazione_manuale"] = ""

    # Selezione colonne di output (tollerante a colonne mancanti)
    colonne_presenti = [c for c in COLONNE_OUTPUT if c in df_gold.columns]
    df_gold = df_gold[colonne_presenti]

    # Mescolamento casuale per evitare bias nella fase di annotazione
    df_gold = df_gold.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df_gold["id"] = df_gold.index + 1   # Riassegna ID dopo il shuffle

    # Salvataggio con encoding utf-8-sig (compatibile con Excel su Windows/Mac)
    df_gold.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    log.info("Salvato → %s  (%d righe)", OUTPUT_FILE, len(df_gold))

    # ── Riepilogo a video ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RIEPILOGO CAMPIONAMENTO")
    print("=" * 65)
    print(f"  {'Finestra':<22} {'Tot. raw':>9} {'Italiano':>9} {'Campione':>9}")
    print("  " + "-" * 52)
    tot_camp = 0
    for s in statistiche:
        print(
            f"  {s['finestra']:<22} {s['totale_raw']:>9} "
            f"{s['n_italiano']:>9} {s['n_campionato']:>9}"
        )
        tot_camp += s["n_campionato"]
    print("  " + "-" * 52)
    print(f"  {'TOTALE':<22} {'':>9} {'':>9} {tot_camp:>9}")
    print("=" * 65)
    print(f"\n  File pronto per annotazione manuale:")
    print(f"  → {OUTPUT_FILE.resolve()}")
    print()
    print("  Istruzioni annotazione:")
    print("  Aprire il CSV in Excel e compilare la colonna 'annotazione_manuale'")
    print("  con uno dei due valori ammessi:  positive  /  negative")
    print("  Salvare in formato CSV (UTF-8) prima di eseguire empirical_validation.py")
    print()


if __name__ == "__main__":
    genera_gold_standard()
