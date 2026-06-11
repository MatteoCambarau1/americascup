# CLAUDE.md — Contesto progetto

## Progetto
Analisi dell'impatto turistico dell'America's Cup 2026 a Cagliari attraverso
recensioni scrappate da piattaforme online. Progetto universitario UniCA, DSBAI.

## Struttura repository
```
booking_scraper_olbia.py    # scraper Selenium per Booking.com (Olbia)
analysis/
├── config.py               # finestre temporali, città, sorgenti, path cartelle
├── utils.py                # parse_italian_date(), assign_temporal_window()
├── preprocessing.py        # carica CSV, pulizia testo, lemmatizzazione, feature engineering
├── sentiment_analysis.py   # Feel-IT / VADER / DistilRoBERTa, aggregazioni
├── topic_modeling.py       # LDA + BERTopic, GridSearch, pyLDAvis, WordCloud
├── comparative.py          # analisi comparativa finale, summary_report.txt
└── validation.py           # valida i modelli NLP su benchmark pubblici
data/
├── raw/                    # CSV grezzi scrappati (input pipeline)
├── processed/              # CSV processati (output preprocessing/sentiment/topics)
└── baselines/              # (riservato per dati baseline 2025)
results/                    # output finali (aggregazioni, grafici, report)
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

## Scraper esistente (booking_scraper_olbia.py)
- Piattaforma: Booking.com
- Città configurata: **Olbia** (manca ancora Cagliari)
- Stato: **scraping in corso** — non tutti i CSV sono ancora disponibili
- Colonne output scraper: `city, source, property, property_stars, scraped_at,
  review_date, rating, review_title, stay_type, nights_stayed, text_positive,
  text_negative, reviewer, reviewer_country`
- **Differenze rispetto allo schema atteso**: `city`→`property_city`,
  `source`→`scrape_source`, `property`→`property_name`
- **Decisione presa**: NON rinominare le colonne ora, NON ri-scrapare.
  `preprocessing.py` gestisce silenziosamente le colonne mancanti/diverse.
- `stay_date` non estratta dallo scraper → lasciata NaN, non impatta l'analisi
  (assign_temporal_window usa review_date)

## Ordine di esecuzione pipeline
```bash
# Opzionale ma consigliato — valida i modelli NLP prima del run completo
python -m analysis.validation

# Pipeline principale
python -m analysis.preprocessing
python -m analysis.sentiment_analysis
python -m analysis.topic_modeling
python -m analysis.comparative
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

## Note importanti
- Ottimizzato per **Apple M2 8GB RAM**: batch_size=8, device=-1 (CPU), no MPS
- Tutti gli script supportano `--sample N` per testare su un sottoinsieme
- `topic_modeling.py` ha `--no-gridsearch` per saltare GridSearch (più veloce)
- I modelli HuggingFace usati: `MilaNLProc/feel-it-italian-sentiment`,
  `MilaNLProc/feel-it-italian-emotion`, `j-hartmann/emotion-english-distilroberta-base`
- BERTopic usa `paraphrase-multilingual-MiniLM-L12-v2` (multilingua, ~120MB)
