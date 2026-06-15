"""
Validazione empirica di Feel-IT e nlptown/bert-base-multilingual-uncased-sentiment
sul gold standard annotato manualmente.

Carica validation/gold_standard_annotato.csv (compilato a mano dopo
generate_gold_standard.py), applica entrambi i modelli e confronta le predizioni
con le etichette umane, producendo metriche e grafici in validation/outputs/.

Utilizzo
--------
    python -m analysis.empirical_validation

    # Se il file annotato ha un nome diverso:
    python -m analysis.empirical_validation --gold validation/mio_file.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from analysis.plot_style import apply_style

apply_style()

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

VALIDATION_DIR = Path("validation")
OUTPUT_DIR     = VALIDATION_DIR / "outputs"

# File di default atteso dopo l'annotazione manuale
DEFAULT_GOLD_FILE = VALIDATION_DIR / "gold_standard_annotato.csv"

# Modello Feel-IT per il sentiment in italiano
MODEL_FEEL_IT = "MilaNLProc/feel-it-italian-sentiment"

# Modello nlptown multilingua (addestrato su recensioni alberghiere)
MODEL_NLPTOWN = "nlptown/bert-base-multilingual-uncased-sentiment"

# Dimensione batch: 16 è conservativo per M2 8 GB su CPU
BATCH_SIZE = 16

# Seed per riproducibilità
RANDOM_STATE = 42

# Valori ammessi per la colonna annotazione_manuale
ETICHETTE_VALIDE = {"positive", "negative"}

# ---------------------------------------------------------------------------
# Patch Feel-IT (stessa logica di sentiment_analysis.py)
# ---------------------------------------------------------------------------

def _patch_camembert_tokenizer() -> None:
    """
    Monkey-patch runtime del tokenizzatore CamemBERT per compatibilità
    con transformers ≥5.0 (il vocabolario SentencePiece torna 3-tuple
    anziché 2-tuple; qui normalizziamo a 2-tuple prima del caricamento).
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


def _ensure_feel_it_tokenizer() -> None:
    """
    Copia il tokenizer.json corretto nella cache HuggingFace se quello
    presente non ha il campo 'type: Unigram' richiesto da transformers ≥5.0.
    """
    patch_file = Path("analysis/patches/feel_it_tokenizer.json")
    if not patch_file.exists():
        return
    try:
        import json
        import shutil
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(MODEL_FEEL_IT, "tokenizer.json")
        with open(cached) as f:
            data = json.load(f)
        if data.get("model", {}).get("type") != "Unigram":
            shutil.copy(patch_file, cached)
            log.info("Patch tokenizer Feel-IT applicata → %s", cached)
    except Exception:
        pass


def _applica_patch_feel_it() -> None:
    """Applica entrambe le patch prima del caricamento del modello."""
    _patch_camembert_tokenizer()
    _ensure_feel_it_tokenizer()


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class FeelItValidator:
    """
    Valida Feel-IT sul gold standard annotato manualmente.

    Parametri
    ---------
    gold_file : Path al CSV annotato (default: validation/gold_standard_annotato.csv)
    batch_size : Dimensione batch per l'inferenza (default: 16)
    """

    def __init__(
        self,
        gold_file: Path = DEFAULT_GOLD_FILE,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.gold_file  = Path(gold_file)
        self.batch_size = batch_size
        self.pipeline_  = None   # Inizializzato in _carica_modello()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Caricamento dati ────────────────────────────────────────────────

    def _carica_gold_standard(self) -> pd.DataFrame:
        """
        Carica il CSV annotato e verifica la colonna annotazione_manuale.
        Le righe con annotazione mancante o non valida vengono escluse.
        """
        if not self.gold_file.exists():
            raise FileNotFoundError(
                f"File gold standard non trovato: {self.gold_file}\n"
                "Esegui prima: python -m analysis.generate_gold_standard\n"
                "poi annota manualmente la colonna 'annotazione_manuale' "
                "e salva il file come gold_standard_annotato.csv"
            )

        df = pd.read_csv(self.gold_file, encoding="utf-8-sig", low_memory=False)
        log.info("Caricate %d righe da %s", len(df), self.gold_file)

        # ── Verifica colonna annotazione ────────────────────────────────
        if "annotazione_manuale" not in df.columns:
            raise ValueError(
                "Colonna 'annotazione_manuale' assente. "
                "Assicurati di aver compilato il file prima di eseguire la validazione."
            )

        # Normalizza a minuscolo e stringa
        df["annotazione_manuale"] = (
            df["annotazione_manuale"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Righe non annotate (vuote, nan, 'nan')
        righe_vuote = df["annotazione_manuale"].isin(["", "nan", "none"])
        n_vuote = righe_vuote.sum()
        if n_vuote > 0:
            log.warning(
                "%d righe con annotazione mancante escluse dalla validazione.", n_vuote
            )
            df = df[~righe_vuote].copy()

        # Righe con etichette non riconosciute
        etichette_non_valide = ~df["annotazione_manuale"].isin(ETICHETTE_VALIDE)
        n_invalide = etichette_non_valide.sum()
        if n_invalide > 0:
            esempi = df.loc[etichette_non_valide, "annotazione_manuale"].unique()[:5]
            log.warning(
                "%d righe con etichette non valide escluse. Esempi: %s",
                n_invalide, list(esempi),
            )
            df = df[~etichette_non_valide].copy()

        log.info("%d righe valide per la validazione.", len(df))
        return df.reset_index(drop=True)

    # ── Modello Feel-IT ─────────────────────────────────────────────────

    def _carica_modello(self) -> None:
        """
        Applica le patch di compatibilità e carica la pipeline Feel-IT.
        Usa sempre CPU (device=-1) per stabilità su Apple M2.
        """
        from transformers import pipeline as hf_pipeline

        log.info("Applicazione patch Feel-IT per transformers ≥5.0 …")
        _applica_patch_feel_it()

        log.info("Caricamento modello: %s …", MODEL_FEEL_IT)
        self.pipeline_ = hf_pipeline(
            "text-classification",
            model=MODEL_FEEL_IT,
            device=-1,          # CPU — no CUDA, no MPS
            truncation=True,
            max_length=512,
            top_k=1,
        )
        log.info("Modello caricato.")

    # ── Inferenza ───────────────────────────────────────────────────────

    def _predici_in_batch(self, testi: list[str]) -> list[str]:
        """
        Esegue l'inferenza in batch e normalizza le etichette restituite
        da Feel-IT verso 'positive' / 'negative'.
        """
        predizioni: list[str] = []

        for i in tqdm(
            range(0, len(testi), self.batch_size),
            desc="Inferenza Feel-IT",
            unit="batch",
        ):
            batch = testi[i : i + self.batch_size]
            # Sostituisce testi vuoti con uno spazio (evita crash pipeline)
            batch = [t if isinstance(t, str) and t.strip() else " " for t in batch]

            risultati = self.pipeline_(batch)

            for pred in risultati:
                # top_k=1 può restituire lista di dict o dict singolo
                if isinstance(pred, list):
                    pred = pred[0]
                etichetta = pred["label"].lower().strip()
                # Normalizzazione etichette Feel-IT
                if etichetta in ("positive", "label_1"):
                    predizioni.append("positive")
                else:
                    predizioni.append("negative")

        return predizioni

    # ── Grafici ─────────────────────────────────────────────────────────

    def _salva_confusion_matrix(
        self,
        y_true: list[str],
        y_pred: list[str],
    ) -> str:
        """Heatmap della confusion matrix con etichette in italiano."""
        etichette = ["positive", "negative"]
        cm = confusion_matrix(y_true, y_pred, labels=etichette)

        fig, ax = plt.subplots(figsize=(6, 4.6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=0.6,
            linecolor="white",
            annot_kws={"fontsize": 13, "fontweight": "bold"},
            xticklabels=["Previsto Positivo", "Previsto Negativo"],
            yticklabels=["Reale Positivo", "Reale Negativo"],
            ax=ax,
        )
        ax.set_title("Confusion Matrix — Feel-IT vs. Annotazione Umana", pad=14, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predizione modello")
        ax.set_ylabel("Annotazione manuale")
        ax.tick_params(axis="both", length=0)
        plt.tight_layout()

        path_out = OUTPUT_DIR / "confusion_matrix_feelit.png"
        fig.savefig(path_out, dpi=150)
        plt.close(fig)
        log.info("Salvata confusion matrix → %s", path_out)
        return str(path_out)

    def _salva_accuracy_per_finestra(
        self,
        df: pd.DataFrame,
    ) -> Optional[str]:
        """
        Grafico a barre con l'accuracy di Feel-IT separata per finestra
        temporale (utile per capire se l'evento influisce sulla qualità
        del modello su questo tipo di testo).
        """
        if "temporal_window" not in df.columns:
            log.warning("Colonna 'temporal_window' assente — grafico per finestra saltato.")
            return None

        finestre = df["temporal_window"].unique()
        acc_per_finestra = {}
        n_per_finestra   = {}

        for finestra in sorted(finestre):
            sotto = df[df["temporal_window"] == finestra]
            if len(sotto) == 0:
                continue
            acc_per_finestra[finestra] = accuracy_score(
                sotto["annotazione_manuale"], sotto["predizione"]
            )
            n_per_finestra[finestra] = len(sotto)

        if not acc_per_finestra:
            return None

        etichette = list(acc_per_finestra.keys())
        valori    = [acc_per_finestra[f] for f in etichette]
        conteggi  = [n_per_finestra[f] for f in etichette]

        colori = ["#127EEA" if "baseline" not in e else "#94A3B8" for e in etichette]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(etichette, valori, color=colori, alpha=0.88, width=0.55)

        # Annotazione: accuracy + numero di campioni
        for bar, acc, n in zip(bars, valori, conteggi):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.2%}\n(n={n})",
                ha="center", va="bottom", fontsize=9,
            )

        ax.axhline(
            sum(valori) / len(valori),
            ls="--", color="#F5C518", lw=1.5, label="Media complessiva",
        )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Feel-IT per finestra temporale", fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)
        ax.legend(fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.yaxis.grid(True, color="#e8edf2", lw=0.8)
        ax.set_axisbelow(True)
        plt.tight_layout()

        path_out = OUTPUT_DIR / "accuracy_per_finestra.png"
        fig.savefig(path_out, dpi=150)
        plt.close(fig)
        log.info("Salvato grafico accuracy per finestra → %s", path_out)
        return str(path_out)

    # ── Metodi pubblici ─────────────────────────────────────────────────

    def valida(self) -> pd.DataFrame:
        """
        Esegue l'intera pipeline di validazione empirica e restituisce
        un DataFrame con i risultati per ogni recensione.
        """
        # 1. Caricamento dati annotati
        df = self._carica_gold_standard()

        # 2. Caricamento modello Feel-IT
        self._carica_modello()

        # 3. Inferenza batch su text_positive
        testi = df["text_positive"].fillna("").tolist()
        log.info("Avvio inferenza su %d testi (batch_size=%d) …", len(testi), self.batch_size)
        df["predizione"] = self._predici_in_batch(testi)

        # 4. Metriche complessive
        y_true = df["annotazione_manuale"].tolist()
        y_pred = df["predizione"].tolist()
        etichette = sorted(ETICHETTE_VALIDE)

        acc = accuracy_score(y_true, y_pred)
        report_dict = classification_report(
            y_true, y_pred, labels=etichette, zero_division=0, output_dict=True
        )
        report_str  = classification_report(
            y_true, y_pred, labels=etichette, zero_division=0
        )

        # 5. Salvataggio confusion matrix
        self._salva_confusion_matrix(y_true, y_pred)

        # 6. Salvataggio classification report come CSV
        report_df = pd.DataFrame(report_dict).transpose().round(4)
        report_path = OUTPUT_DIR / "classification_report_feelit.csv"
        report_df.to_csv(report_path, encoding="utf-8-sig")
        log.info("Salvato classification report → %s", report_path)

        # 7. Accuracy per finestra temporale
        self._salva_accuracy_per_finestra(df)

        # 8. Stampa riepilogo a video
        self._stampa_riepilogo(acc, report_str, report_dict, df)

        return df

    def _stampa_riepilogo(
        self,
        acc: float,
        report_str: str,
        report_dict: dict,
        df: pd.DataFrame,
    ) -> None:
        """Stampa un riepilogo leggibile dei risultati della validazione."""
        print("\n" + "=" * 65)
        print("  RISULTATI VALIDAZIONE EMPIRICA — Feel-IT vs. Annotazione Umana")
        print("=" * 65)
        print(f"\n  Accuracy complessiva : {acc:.4f}  ({acc * 100:.1f}%)")

        for classe in sorted(ETICHETTE_VALIDE):
            if classe in report_dict:
                r = report_dict[classe]
                print(
                    f"  {classe:<12}  "
                    f"Precision={r['precision']:.3f}  "
                    f"Recall={r['recall']:.3f}  "
                    f"F1={r['f1-score']:.3f}  "
                    f"(n={int(r['support'])})"
                )

        print(f"\n  Macro F1    : {report_dict['macro avg']['f1-score']:.4f}")
        print(f"  Weighted F1 : {report_dict['weighted avg']['f1-score']:.4f}")

        print("\n" + "-" * 65)
        print(f"  {'Finestra':<22}  {'N':>5}  {'Accuracy':>10}")
        print("  " + "-" * 42)

        if "temporal_window" in df.columns:
            for finestra in sorted(df["temporal_window"].unique()):
                sotto = df[df["temporal_window"] == finestra]
                acc_f = accuracy_score(
                    sotto["annotazione_manuale"], sotto["predizione"]
                )
                print(f"  {finestra:<22}  {len(sotto):>5}  {acc_f:>10.2%}")

        print("\n" + "=" * 65)
        print(f"  Output salvati in: {OUTPUT_DIR.resolve()}")
        print("    - confusion_matrix_feelit.png")
        print("    - classification_report_feelit.csv")
        print("    - accuracy_per_finestra.png")
        print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Classe NlpTownValidator
# ---------------------------------------------------------------------------

class NlpTownValidator:
    """
    Valida nlptown/bert-base-multilingual-uncased-sentiment sullo stesso
    gold standard usato da FeelItValidator.

    Il modello restituisce stelle 1-5:
        1-2 stelle → negative
        3-4-5 stelle → positive  (mapping conservativo: il 3 va a positive)
    """

    # Mappa etichette del modello → positive/negative
    _MAPPA_STELLE: dict[str, str] = {
        "1 star":  "negative",
        "2 stars": "negative",
        "3 stars": "positive",
        "4 stars": "positive",
        "5 stars": "positive",
    }

    def __init__(
        self,
        gold_file: Path = DEFAULT_GOLD_FILE,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.gold_file  = Path(gold_file)
        self.batch_size = batch_size
        self.pipeline_  = None

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Caricamento dati (riusa la stessa logica di FeelItValidator) ─────

    def _carica_gold_standard(self) -> pd.DataFrame:
        """Carica e verifica il CSV annotato manualmente."""
        if not self.gold_file.exists():
            raise FileNotFoundError(
                f"File gold standard non trovato: {self.gold_file}\n"
                "Esegui prima: python -m analysis.generate_gold_standard\n"
                "poi annota manualmente la colonna 'annotazione_manuale'."
            )

        df = pd.read_csv(self.gold_file, encoding="utf-8-sig", low_memory=False)
        log.info("Caricate %d righe da %s", len(df), self.gold_file)

        if "annotazione_manuale" not in df.columns:
            raise ValueError("Colonna 'annotazione_manuale' assente nel gold standard.")

        df["annotazione_manuale"] = (
            df["annotazione_manuale"].astype(str).str.strip().str.lower()
        )

        # Escludi righe non annotate
        righe_vuote = df["annotazione_manuale"].isin(["", "nan", "none"])
        if righe_vuote.sum() > 0:
            log.warning("%d righe non annotate escluse.", righe_vuote.sum())
            df = df[~righe_vuote].copy()

        # Escludi etichette non riconosciute
        non_valide = ~df["annotazione_manuale"].isin(ETICHETTE_VALIDE)
        if non_valide.sum() > 0:
            log.warning("%d righe con etichette non valide escluse.", non_valide.sum())
            df = df[~non_valide].copy()

        log.info("%d righe valide per la validazione.", len(df))
        return df.reset_index(drop=True)

    # ── Caricamento modello ──────────────────────────────────────────────

    def _carica_modello(self) -> None:
        """Carica la pipeline nlptown su CPU (device=-1)."""
        from transformers import pipeline as hf_pipeline

        log.info("Caricamento modello: %s …", MODEL_NLPTOWN)
        self.pipeline_ = hf_pipeline(
            "text-classification",
            model=MODEL_NLPTOWN,
            device=-1,        # CPU — no CUDA, no MPS
            truncation=True,
            max_length=512,
            top_k=1,
        )
        log.info("Modello caricato.")

    # ── Inferenza ───────────────────────────────────────────────────────

    def _predici_in_batch(self, testi: list[str]) -> list[str]:
        """
        Inferenza batch: converte le stelle restituite da nlptown
        in positive/negative secondo _MAPPA_STELLE.
        """
        predizioni: list[str] = []

        for i in tqdm(
            range(0, len(testi), self.batch_size),
            desc="Inferenza nlptown",
            unit="batch",
        ):
            batch = testi[i : i + self.batch_size]
            batch = [t if isinstance(t, str) and t.strip() else " " for t in batch]

            risultati = self.pipeline_(batch)

            for pred in risultati:
                if isinstance(pred, list):
                    pred = pred[0]
                etichetta_stelle = pred["label"].lower().strip()
                # Mappa stelle → positive/negative (default: positive)
                predizioni.append(self._MAPPA_STELLE.get(etichetta_stelle, "positive"))

        return predizioni

    # ── Grafici ─────────────────────────────────────────────────────────

    def _salva_confusion_matrix(self, y_true: list[str], y_pred: list[str]) -> None:
        """Heatmap della confusion matrix con etichette in italiano."""
        etichette = ["positive", "negative"]
        cm = confusion_matrix(y_true, y_pred, labels=etichette)

        fig, ax = plt.subplots(figsize=(6, 4.6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            cbar=False,
            linewidths=0.6,
            linecolor="white",
            annot_kws={"fontsize": 13, "fontweight": "bold"},
            xticklabels=["Previsto Positivo", "Previsto Negativo"],
            yticklabels=["Reale Positivo", "Reale Negativo"],
            ax=ax,
        )
        ax.set_title("Confusion Matrix — nlptown vs. Annotazione Umana", pad=14, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predizione modello")
        ax.set_ylabel("Annotazione manuale")
        ax.tick_params(axis="both", length=0)
        plt.tight_layout()

        path_out = OUTPUT_DIR / "confusion_matrix_nlptown.png"
        fig.savefig(path_out, dpi=150)
        plt.close(fig)
        log.info("Salvata confusion matrix → %s", path_out)

    def _salva_accuracy_per_finestra(self, df: pd.DataFrame) -> None:
        """Grafico a barre dell'accuracy nlptown per finestra temporale."""
        if "temporal_window" not in df.columns:
            log.warning("Colonna 'temporal_window' assente — grafico per finestra saltato.")
            return

        acc_per_finestra = {}
        n_per_finestra   = {}
        for finestra in sorted(df["temporal_window"].unique()):
            sotto = df[df["temporal_window"] == finestra]
            if len(sotto) == 0:
                continue
            acc_per_finestra[finestra] = accuracy_score(
                sotto["annotazione_manuale"], sotto["predizione"]
            )
            n_per_finestra[finestra] = len(sotto)

        if not acc_per_finestra:
            return

        etichette = list(acc_per_finestra.keys())
        valori    = [acc_per_finestra[f] for f in etichette]
        conteggi  = [n_per_finestra[f] for f in etichette]
        colori    = ["#E8650A" if "baseline" not in e else "#F5BE8A" for e in etichette]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(etichette, valori, color=colori, alpha=0.88, width=0.55)

        for bar, acc, n in zip(bars, valori, conteggi):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.2%}\n(n={n})",
                ha="center", va="bottom", fontsize=9,
            )

        ax.axhline(
            sum(valori) / len(valori),
            ls="--", color="#127EEA", lw=1.5, label="Media complessiva",
        )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy nlptown per finestra temporale", fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)
        ax.legend(fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.yaxis.grid(True, color="#e8edf2", lw=0.8)
        ax.set_axisbelow(True)
        plt.tight_layout()

        path_out = OUTPUT_DIR / "accuracy_per_finestra_nlptown.png"
        fig.savefig(path_out, dpi=150)
        plt.close(fig)
        log.info("Salvato grafico accuracy per finestra → %s", path_out)

    # ── Metodi pubblici ─────────────────────────────────────────────────

    def valida(self) -> pd.DataFrame:
        """
        Pipeline completa: carica dati, inferenza, metriche, grafici.
        Restituisce il DataFrame con predizioni aggiunte.
        """
        df = self._carica_gold_standard()
        self._carica_modello()

        testi = df["text_positive"].fillna("").tolist()
        log.info("Avvio inferenza su %d testi (batch_size=%d) …", len(testi), self.batch_size)
        df["predizione"] = self._predici_in_batch(testi)

        y_true    = df["annotazione_manuale"].tolist()
        y_pred    = df["predizione"].tolist()
        etichette = sorted(ETICHETTE_VALIDE)

        acc         = accuracy_score(y_true, y_pred)
        report_dict = classification_report(
            y_true, y_pred, labels=etichette, zero_division=0, output_dict=True
        )
        report_str  = classification_report(
            y_true, y_pred, labels=etichette, zero_division=0
        )

        self._salva_confusion_matrix(y_true, y_pred)

        report_df = pd.DataFrame(report_dict).transpose().round(4)
        report_path = OUTPUT_DIR / "classification_report_nlptown.csv"
        report_df.to_csv(report_path, encoding="utf-8-sig")
        log.info("Salvato classification report → %s", report_path)

        self._salva_accuracy_per_finestra(df)
        self._stampa_riepilogo(acc, report_str, report_dict, df)

        return df

    def _stampa_riepilogo(
        self,
        acc: float,
        report_str: str,
        report_dict: dict,
        df: pd.DataFrame,
    ) -> None:
        """Stampa un riepilogo leggibile dei risultati."""
        print("\n" + "=" * 65)
        print("  RISULTATI VALIDAZIONE EMPIRICA — nlptown vs. Annotazione Umana")
        print("=" * 65)
        print(f"\n  Accuracy complessiva : {acc:.4f}  ({acc * 100:.1f}%)")

        for classe in sorted(ETICHETTE_VALIDE):
            if classe in report_dict:
                r = report_dict[classe]
                print(
                    f"  {classe:<12}  "
                    f"Precision={r['precision']:.3f}  "
                    f"Recall={r['recall']:.3f}  "
                    f"F1={r['f1-score']:.3f}  "
                    f"(n={int(r['support'])})"
                )

        print(f"\n  Macro F1    : {report_dict['macro avg']['f1-score']:.4f}")
        print(f"  Weighted F1 : {report_dict['weighted avg']['f1-score']:.4f}")

        print("\n" + "-" * 65)
        print(f"  {'Finestra':<22}  {'N':>5}  {'Accuracy':>10}")
        print("  " + "-" * 42)

        if "temporal_window" in df.columns:
            for finestra in sorted(df["temporal_window"].unique()):
                sotto = df[df["temporal_window"] == finestra]
                acc_f = accuracy_score(sotto["annotazione_manuale"], sotto["predizione"])
                print(f"  {finestra:<22}  {len(sotto):>5}  {acc_f:>10.2%}")

        print("\n" + "=" * 65)
        print(f"  Output salvati in: {OUTPUT_DIR.resolve()}")
        print("    - confusion_matrix_nlptown.png")
        print("    - classification_report_nlptown.csv")
        print("    - accuracy_per_finestra_nlptown.png")
        print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Confronto tra modelli
# ---------------------------------------------------------------------------

def _stampa_confronto(risultati_feelit: dict, risultati_nlptown: dict) -> None:
    """
    Stampa e salva la tabella comparativa tra Feel-IT e nlptown.
    """
    righe = [
        {
            "modello":      "Feel-IT (MilaNLProc)",
            "accuracy":     round(risultati_feelit["accuracy"], 4),
            "macro_f1":     round(risultati_feelit["macro avg"]["f1-score"], 4),
            "weighted_f1":  round(risultati_feelit["weighted avg"]["f1-score"], 4),
        },
        {
            "modello":      "nlptown (bert-multilingual)",
            "accuracy":     round(risultati_nlptown["accuracy"], 4),
            "macro_f1":     round(risultati_nlptown["macro avg"]["f1-score"], 4),
            "weighted_f1":  round(risultati_nlptown["weighted avg"]["f1-score"], 4),
        },
    ]

    df_confronto = pd.DataFrame(righe)

    # Salva CSV
    path_csv = OUTPUT_DIR / "confronto_modelli.csv"
    df_confronto.to_csv(path_csv, index=False, encoding="utf-8-sig")
    log.info("Salvato confronto modelli → %s", path_csv)

    # Stampa a video
    print("\n" + "=" * 65)
    print("  CONFRONTO MODELLI")
    print("=" * 65)
    print(f"\n  {'Modello':<30}  {'Accuracy':>10}  {'Macro F1':>10}  {'Weighted F1':>12}")
    print("  " + "-" * 58)
    for _, row in df_confronto.iterrows():
        print(
            f"  {row['modello']:<30}  "
            f"{row['accuracy']:>10.4f}  "
            f"{row['macro_f1']:>10.4f}  "
            f"{row['weighted_f1']:>12.4f}"
        )
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Entry point CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validazione empirica di Feel-IT e nlptown sul gold standard annotato.\n"
            "Assicurati di aver compilato il gold standard prima di eseguire."
        )
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_FILE,
        help=f"Path al CSV annotato (default: {DEFAULT_GOLD_FILE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Dimensione batch per l'inferenza (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--only",
        choices=["feelit", "nlptown"],
        default=None,
        help="Esegui solo uno dei due modelli (default: entrambi)",
    )
    args = parser.parse_args()

    try:
        risultati_feelit   = None
        risultati_nlptown  = None

        # ── 1. Feel-IT ───────────────────────────────────────────────────
        if args.only in (None, "feelit"):
            validator_feelit = FeelItValidator(
                gold_file=args.gold,
                batch_size=args.batch_size,
            )
            df_feelit = validator_feelit.valida()

            # Raccoglie metriche per il confronto finale
            y_true = df_feelit["annotazione_manuale"].tolist()
            y_pred = df_feelit["predizione"].tolist()
            risultati_feelit = classification_report(
                y_true, y_pred,
                labels=sorted(ETICHETTE_VALIDE),
                zero_division=0,
                output_dict=True,
            )
            risultati_feelit["accuracy"] = accuracy_score(y_true, y_pred)

        # ── 2. nlptown ───────────────────────────────────────────────────
        if args.only in (None, "nlptown"):
            validator_nlptown = NlpTownValidator(
                gold_file=args.gold,
                batch_size=args.batch_size,
            )
            df_nlptown = validator_nlptown.valida()

            y_true = df_nlptown["annotazione_manuale"].tolist()
            y_pred = df_nlptown["predizione"].tolist()
            risultati_nlptown = classification_report(
                y_true, y_pred,
                labels=sorted(ETICHETTE_VALIDE),
                zero_division=0,
                output_dict=True,
            )
            risultati_nlptown["accuracy"] = accuracy_score(y_true, y_pred)

        # ── 3. Confronto finale ──────────────────────────────────────────
        if risultati_feelit is not None and risultati_nlptown is not None:
            _stampa_confronto(risultati_feelit, risultati_nlptown)

    except FileNotFoundError as e:
        print(f"\n[ERRORE] {e}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
