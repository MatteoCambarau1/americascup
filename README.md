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
│   ├── raw/                      # CSV grezzi estratti dagli scraper
│   ├── processed/                # CSV processati dalla pipeline
│   └── baselines/                # Dati baseline anno precedente (2025)
│
├── analysis/
│   ├── config.py                 # Finestre temporali, città, path cartelle
│   ├── utils.py                  # Parsing date italiane, assegnazione finestre
│   ├── preprocessing.py          # Pulizia testo, lemmatizzazione, feature engineering
│   ├── sentiment_analysis.py     # Sentiment & Emotion Analysis (Feel-IT / VADER)
│   ├── topic_modeling.py         # LDA + BERTopic, GridSearch, pyLDAvis, WordCloud
│   ├── comparative.py            # Analisi comparativa finale, summary report
│   └── validation.py             # Validazione modelli NLP su benchmark pubblici
│
├── results/                      # Output grafici, aggregazioni, report
│
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

### 2. Validazione modelli NLP (opzionale ma consigliato)

```bash
python -m analysis.validation
python -m analysis.validation --max-samples 50   # test rapido
```

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

---

## 🔧 Pipeline Completa

```
Fase 1: Estrazione Dati
    ├── booking_scraper_cagliari.py
    └── booking_scraper_olbia.py

Fase 2: Validazione Modelli (opzionale)
    └── analysis/validation.py

Fase 3: Preprocessing
    └── analysis/preprocessing.py

Fase 4: Analisi NLP
    ├── analysis/sentiment_analysis.py   (Feel-IT / VADER / DistilRoBERTa)
    └── analysis/topic_modeling.py       (LDA + BERTopic)

Fase 5: Analisi Comparativa
    └── analysis/comparative.py
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

- **Report tecnico** – documento scritto con metodologia, risultati e conclusioni
- **Presentazione PowerPoint** – sintesi visiva dei risultati per la presentazione d'esame

---

## 🧪 Analisi Aggiuntive (Opzionali)

- **PCA** – riduzione dimensionale per visualizzare cluster di topic/sentimenti
- **SVM** – classificazione supervisionata del sentiment
- Integrazione con interviste sul campo (pre/post evento) per validazione

---

## 👤 Autore

Progetto universitario Matteo Cambarau – Esame di Web Analytics e Analisi Testuale

Supervisor: *Professor Marco Ortu*  
Anno Accademico: 2025/2026

---

## 📄 Licenza

Questo progetto è sviluppato a scopo accademico. I dati estratti sono utilizzati esclusivamente per ricerca e non vengono redistribuiti.
