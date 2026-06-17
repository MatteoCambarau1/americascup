# 🏆 America's Cup Cagliari – Tourist Impact Analysis

> Analisi dell'impatto turistico dell'America's Cup 2026 a Cagliari attraverso text mining, topic modeling e sentiment analysis su dati estratti da piattaforme online.

---

## 📋 Descrizione del Progetto

Il progetto mira a misurare e quantificare l'impatto dell'**America's Cup 2026** sul turismo a Cagliari, analizzando recensioni e contenuti testuali estratti da piattaforme online in tre finestre temporali distinte. I risultati vengono confrontati con la baseline dell'anno precedente e con una città di controllo (**Olbia**).

---

## 🗓️ Finestre Temporali

| Fase               | Periodo                          |
|--------------------|----------------------------------|
| Pre-evento         | 1 aprile – 20 maggio 2026        |
| Durante evento     | 21 maggio – 24 maggio 2026       |
| Post-evento        | 25 maggio – 15 giugno 2026       |
| Baseline 2025      | Stesse finestre, anno precedente |
| Città di controllo | Olbia (stessa finestra, nessun evento) |

---

## 🌐 Fonti Dati

- **Booking.com** – recensioni alloggi (scraper Selenium)

---

## 📂 Struttura del Repository

```
project/
│
├── booking_scraper_cagliari.py   # Scraper Booking.com per Cagliari
├── booking_scraper_olbia.py      # Scraper Booking.com per Olbia
│
├── data/
│   ├── raw/                      # CSV grezzi estratti dagli scraper (dati 2026 e baseline 2025)
│   └── processed/                # CSV processati dalla pipeline
│
├── analysis/
│   ├── config.py                 # Finestre temporali, città, path cartelle
│   ├── utils.py                  # Parsing date italiane, assegnazione finestre
│   ├── preprocessing.py          # Pulizia testo, lemmatizzazione, feature engineering
│   ├── sentiment_analysis.py     # Sentiment & Emotion Analysis (nlptown / Feel-IT emotion / DistilRoBERTa)
│   ├── topic_modeling.py         # LDA + BERTopic, GridSearch, pyLDAvis, WordCloud
│   ├── comparative.py            # Analisi comparativa finale, summary report
│   ├── exploratory.py            # Analisi temporale descrittiva (box plot, 2026 vs 2025)
│   ├── validation.py             # Validazione modelli NLP su benchmark esterni — ESEGUITA
│   ├── generate_gold_standard.py # Genera il gold standard per la validazione in-domain
│   ├── empirical_validation.py   # Valida Feel-IT vs nlptown sul gold standard annotato
│   └── patches/                  # Fix tokenizer Feel-IT (transformers 5.x)
│
├── validation/                   # Gold standard e output della validazione in-domain
├── results/                      # Output grafici, aggregazioni, report
│
├── report_tecnico.md             # Report tecnico completo (fonte canonica)
├── report_tecnico.docx           # Report tecnico in formato Word
├── CLAUDE.md                     # Contesto progetto per Claude Code
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installazione

### Prerequisiti
- Python 3.9+
- Google Chrome + ChromeDriver (per Selenium)

### Installazione

```bash
git clone https://github.com/your-username/americas-cup-cagliari
cd americas-cup-cagliari
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dipendenze principali

```txt
selenium
beautifulsoup4
pandas
nltk
spacy
transformers
datasets
bertopic
sentence-transformers
scikit-learn
vaderSentiment
langdetect
textblob
tqdm
pyldavis
wordcloud
matplotlib
seaborn
```

---

## 🚀 Utilizzo

### 1. Eseguire gli scraper

```bash
python booking_scraper_cagliari.py
python booking_scraper_olbia.py
```

> ⚠️ **Nota**: Modificare `DATE_FROM` e `DATE_TO` nello script per cambiare il range di raccolta. Gli scraper supportano il checkpoint automatico: se interrotti, riprendono dalla struttura successiva.

### 2. Validazione modelli NLP (eseguita)

```bash
python -m analysis.validation
python -m analysis.validation --max-samples 50   # test rapido
```

> ℹ️ Due validazioni sono state eseguite per il report:
> - **In-domain** (sezione 9): `analysis/empirical_validation.py` confronta Feel-IT vs
>   `nlptown` su 150 recensioni annotate manualmente. Risultato: nlptown accuracy 96%
>   vs Feel-IT 58% → nlptown adottato come modello principale.
> - **Esterna** (sezione 9.4): `analysis/validation.py` valida nlptown su
>   `mteb/tweet_sentiment_multilingual` (IT: acc 65.2%, EN: acc 70.4%) e Feel-IT emotion
>   su `dair-ai/emotion` (proxy EN, risultati non conclusivi per mismatch di dominio).

### 3. Preprocessing

```bash
python -m analysis.preprocessing
python -m analysis.preprocessing --sample 100   # test rapido
```

### 4. Sentiment & Emotion Analysis

```bash
python -m analysis.sentiment_analysis
python -m analysis.sentiment_analysis --sample 100   # test rapido
```

### 5. Topic Modeling

```bash
python -m analysis.topic_modeling
python -m analysis.topic_modeling --sample 200 --no-gridsearch   # test rapido
```

### 6. Analisi Comparativa

```bash
python -m analysis.comparative
```

### 7. Analisi Temporale Descrittiva

```bash
python -m analysis.exploratory
```

---

## 🔧 Pipeline Completa

```
Fase 1: Estrazione Dati
    ├── booking_scraper_cagliari.py
    └── booking_scraper_olbia.py

Fase 2: Validazione Modelli (eseguita)
    └── analysis/validation.py

Fase 3: Preprocessing
    └── analysis/preprocessing.py

Fase 4: Analisi NLP
    ├── analysis/sentiment_analysis.py   (nlptown / Feel-IT emotion / DistilRoBERTa)
    └── analysis/topic_modeling.py       (LDA + BERTopic)

Fase 5: Analisi Comparativa
    └── analysis/comparative.py

Fase 6: Analisi Temporale Descrittiva
    └── analysis/exploratory.py
```

---

## 📊 Output Attesi

- **Distribuzione dei topic** per finestra temporale e città
- **Sentiment score medio** per finestra temporale (2026 vs baseline 2025)
- **Emozioni prevalenti** per periodo (gioia, tristezza, rabbia, paura, ecc.)
- **Variazione volumetrica** delle recensioni rispetto al 2025
- **Confronto Cagliari vs Olbia** per stesso periodo
- **Visualizzazioni**: pyLDAvis interattivo, WordCloud, confusion matrix, log-likelihood curves

## 📝 Deliverable Finali

- **`report_tecnico.md` / `report_tecnico.docx`** – report tecnico completo (11 sezioni) con metodologia, risultati, validazione e conclusioni
- **Presentazione PowerPoint** – sintesi visiva dei risultati per la presentazione d'esame

---

## 🧪 Analisi Aggiuntive (Opzionali)

- Validazione esterna su benchmark pubblici (`analysis/validation.py`) — eseguita, risultati in sezione 9.4 del report
- Integrazione con interviste sul campo (pre/post evento) per validazione

> **Nota**: PCA, K-Means e SVM sono state escluse dal progetto — lo sbilanciamento
> delle classi temporali (pre ≈93k / during ≈6k / post ≈10k) rendeva la
> classificazione non interpretabile (vedi `CLAUDE.md`).

---

## 👤 Autore

Progetto universitario Matteo Cambarau – Esame di Web Analytics e Analisi Testuale

Supervisor: *Professor Marco Ortu*  
Anno Accademico: 2025/2026

---

## 📄 Licenza

Questo progetto è sviluppato a scopo accademico. I dati estratti sono utilizzati esclusivamente per ricerca e non vengono redistribuiti.
