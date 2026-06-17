# CLAUDE.md — Contesto progetto

## Progetto
Analisi dell'impatto turistico dell'America's Cup 2026 a Cagliari attraverso
recensioni scrappate da piattaforme online. Progetto universitario UniCA, DSBAI.

## Struttura repository
```
booking_scraper_olbia.py    # scraper Selenium per Booking.com (Olbia) — COMPLETATO
booking_scraper_cagliari.py # scraper Selenium per Booking.com (Cagliari) — COMPLETATO
analysis/
├── config.py               # finestre temporali, città, sorgenti, path cartelle
├── utils.py                # parse_italian_date(), assign_temporal_window()
├── preprocessing.py        # carica CSV, pulizia testo, lemmatizzazione, feature engineering
├── sentiment_analysis.py   # nlptown (sentiment, tutte le lingue) / Feel-IT emotion (IT) / DistilRoBERTa (EN), aggregazioni
├── topic_modeling.py       # LDA + BERTopic, GridSearch, pyLDAvis, WordCloud
├── comparative.py          # analisi comparativa finale, summary_report.txt
├── exploratory.py          # analisi temporale descrittiva (box plot, confronto 2026 vs 2025)
├── validation.py           # valida i modelli NLP su benchmark pubblici esterni — ESEGUITA (nlptown IT/EN + Feel-IT emotion)
├── generate_gold_standard.py # campiona 150 recensioni Cagliari per annotazione manuale (gold standard)
├── empirical_validation.py # valida Feel-IT e nlptown sul gold standard annotato — ESEGUITO
└── patches/
    └── feel_it_tokenizer.json # fix tokenizer Feel-IT per transformers 5.x
data/
├── raw/                    # CSV grezzi scrappati (9 file, 15509 recensioni totali)
└── processed/              # CSV processati (output preprocessing/sentiment/topics)
validation/
├── gold_standard_da_annotare.csv # template generato da generate_gold_standard.py
└── outputs/                # risultati empirical_validation.py (confusion matrix, classification report, confronto_modelli.csv)
results/                    # output finali (aggregazioni, grafici, report)
report_tecnico.md           # report tecnico completo (11 sezioni) — fonte canonica
report_tecnico.docx         # report tecnico in formato Word, generato da report_tecnico.md
```

## Finestre temporali (config.py)
- **pre**: 1 apr – 20 mag 2026
- **during**: 21 mag – 24 mag 2026
- **post**: 25 mag – 15 giu 2026
- **baseline_pre/during/post**: stesse finestre, anno 2025

## Città
- **Target**: Cagliari
- **Controllo**: Olbia

## Sorgenti pianificate
booking

## Schema colonne dataframe recensioni
`property_name, property_stars, property_city, review_date, stay_date,
reviewer_country, rating, review_title, text_positive, text_negative,
stay_type, nights_stayed, scrape_source`

## Scraper (booking_scraper_olbia.py / booking_scraper_cagliari.py)
- Piattaforma: Booking.com — **scraping completato** per entrambe le città
- CSV disponibili in `data/raw/`: Cagliari (pre/during/post 2025 e 2026), Olbia (pre/during/post 2026)
- Baseline Olbia 2025: **non raccolta** (decisione progettuale)
- Colonne output scraper: `city, source, property, property_stars, scraped_at,
  review_date, rating, review_title, stay_type, nights_stayed, text_positive,
  text_negative, reviewer, reviewer_country`
- **Differenze rispetto allo schema atteso**: `city`→`property_city`,
  `source`→`scrape_source`, `property`→`property_name`
- **Decisione presa**: NON rinominare le colonne ora, NON ri-scrapare.
  La normalizzazione avviene nei singoli script di analisi (alias `city` → `property_city`
  con `.str.title()` per il case matching con config.py).
- `stay_date` non estratta dallo scraper → lasciata NaN, non impatta l'analisi
  (assign_temporal_window usa review_date)

## Stato pipeline (aggiornato 2026-06-17)
Tutti gli step completati con successo su 15509 recensioni.

Validazione empirica in-domain completata (2026-06-15): confronto Feel-IT vs
`nlptown/bert-base-multilingual-uncased-sentiment` su gold standard da 150 recensioni
annotate manualmente. Risultati e interpretazione critica in `report_tecnico.md`,
sezione 9 (Validazione empirica). `analysis/validation.py` (benchmark esterni) eseguita
(2026-06-17): nlptown IT/EN su `mteb/tweet_sentiment_multilingual`, Feel-IT emotion
su `dair-ai/emotion` (proxy EN). Risultati in `results/validation/` e sezione 9.4 del report.

A seguito della validazione, `sentiment_analysis.py` è stato aggiornato (2026-06-17)
per usare `nlptown/bert-base-multilingual-uncased-sentiment` come modello principale
al posto di Feel-IT, in linea con i risultati della validazione empirica.

## Output exploratory.py
Analisi temporale descrittiva — 4 file in `results/exploratory/`:
- `temporal_rating_boxplot.png` — box plot rating per finestra (pre/during/post) × città
- `temporal_sentiment_boxplot.png` — box plot sentiment score per finestra × città
- `temporal_cagliari_2026_vs_2025.png` — confronto bar chart 2026 vs baseline 2025 (rating + sentiment)
- `temporal_summary.csv` — tabella aggregata (n, media, std) per city × window

**Nota**: gli `n` nel CSV sono righe del dataset merged (espanso dal join topic), non recensioni uniche.
PCA, K-Means e SVM sono stati rimossi: class imbalance pre/during/post (93k vs 6k vs 10k)
rendeva la classificazione temporale non interpretabile.

## Validazione empirica (gold standard in-domain)
- `analysis/generate_gold_standard.py` campiona 150 recensioni IT di Cagliari
  (30 per finestra: pre/during/post/baseline_pre/baseline_during, `random_state=42`)
  e scrive `validation/gold_standard_da_annotare.csv`.
- Dopo annotazione manuale (etichetta positive/negative), il file annotato va salvato
  come `validation/gold_standard_annotato.csv` (richiesto da `empirical_validation.py`,
  presente nel repo — 150 righe, 147 positive / 3 negative). Una copia leggibile in
  formato Excel è disponibile in `validation/gold_standard_annotato.xlsx`.
- `analysis/empirical_validation.py` confronta Feel-IT (`MilaNLProc/feel-it-italian-sentiment`)
  e `nlptown/bert-base-multilingual-uncased-sentiment` (mappatura 1-2★→negative, 3-5★→positive)
  sul gold standard, salvando in `validation/outputs/`: confusion matrix, classification
  report e `confronto_modelli.csv` per entrambi i modelli.
- **Risultato chiave**: Feel-IT accuracy 58% / macro F1 0.395; nlptown accuracy 96% / macro F1 0.490.
  Il 96% di nlptown è un artefatto della classe maggioritaria (0/3 negativi identificati
  correttamente): nessuno dei due modelli gestisce bene la classe negativa minoritaria.
  Discussione completa in `report_tecnico.md` sezione 9.2-9.3.

## Ordine di esecuzione pipeline
```bash
# Opzionale ma consigliato — valida i modelli NLP prima del run completo
python -m analysis.validation

# Pipeline principale
python -m analysis.preprocessing
python -m analysis.sentiment_analysis
python -m analysis.topic_modeling --no-gridsearch
python -m analysis.comparative
python -m analysis.exploratory
```

## Dipendenze principali
```bash
pip install pandas nltk spacy transformers datasets
pip install bertopic sentence-transformers scikit-learn
pip install vaderSentiment langdetect textblob tqdm
pip install pyldavis wordcloud matplotlib seaborn
python -m spacy download it_core_news_sm
python -m spacy download en_core_web_sm
```

## Dipendenze aggiuntive installate
```bash
pip install beautifulsoup4 sentencepiece
```

## Note importanti
- Ottimizzato per **Apple M2 8GB RAM**: batch_size=8, device=-1 (CPU), no MPS
- Tutti gli script supportano `--sample N` per testare su un sottoinsieme
- `topic_modeling.py` ha `--no-gridsearch` per saltare GridSearch (più veloce)
- I modelli HuggingFace usati nella pipeline: `nlptown/bert-base-multilingual-uncased-sentiment`
  (sentiment principale, sostituisce Feel-IT), `MilaNLProc/feel-it-italian-emotion`,
  `j-hartmann/emotion-english-distilroberta-base`
- `MilaNLProc/feel-it-italian-sentiment` rimosso dalla pipeline principale dopo la validazione
  empirica (accuracy 58% vs 96% di nlptown); resta referenziato in `empirical_validation.py`
- BERTopic usa `paraphrase-multilingual-MiniLM-L12-v2` (multilingua, ~120MB)

## Bug fix Feel-IT (transformers 5.x) — storico
Feel-IT sentiment è stato rimosso dalla pipeline principale (2026-06-17). Il codice
di patch resta attivo in `sentiment_analysis.py` perché `feel-it-italian-emotion`
(modello emozione, ancora in uso nella pipeline) è anch'esso basato su CamemBERT e
richiede le stesse fix. I patch sono inoltre referenziati da `empirical_validation.py`
per il confronto storico sul gold standard.

Le fix applicate erano:
1. **Monkey-patch CamemBERT** (`_patch_camembert_tokenizer()`): normalizza la vocab da 3-tuple a 2-tuple.
2. **`tokenizer.json` corretto** in `analysis/patches/feel_it_tokenizer.json`, applicato
   automaticamente in cache al primo avvio.

## Normalizzazione colonne city
Il scraper produce `city` (minuscolo) ma `config.py` usa `TARGET_CITY = "Cagliari"` (title case).
La normalizzazione avviene in `comparative.py` con `df["property_city"] = df["city"].str.title()`.
`sentiment_analysis.py` e `topic_modeling.py` gestiscono il fallback `city`/`property_city` internamente.
