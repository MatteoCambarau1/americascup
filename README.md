# 🏆 America's Cup Cagliari – Tourist Impact Analysis

> Analisi dell'impatto turistico dell'America's Cup 2025 a Cagliari attraverso text mining, topic modeling e sentiment analysis su dati estratti da piattaforme online.

---

## 📋 Descrizione del Progetto

Il progetto mira a misurare e quantificare l'impatto dell'**America's Cup 2025** sul turismo a Cagliari, analizzando recensioni e contenuti testuali estratti da piattaforme online in tre finestre temporali distinte. I risultati vengono confrontati con la baseline dell'anno precedente e con una città di controllo (**Olbia**).

---

## 🗓️ Finestre Temporali

| Fase              | Periodo                  |
|-------------------|--------------------------|
| Pre-evento        | 1 aprile – 20 maggio 2025 |
| Durante evento    | 20 maggio – 24 maggio 2025 |
| Post-evento       | 24 maggio – 15 giugno 2025 |
| Baseline 2024     | Stesse finestre, anno precedente |
| Città di controllo | Olbia (stessa finestra, nessun evento) |

---

## 🌐 Fonti Dati

Le fonti sono ordinate per priorità:

### Priorità 1 – Piattaforme turistiche
- **Booking.com** – recensioni alloggi
- **Airbnb** – recensioni alloggi
- **TripAdvisor** – ristoranti, attrazioni, hotel
- **Google Maps** – alloggi, ristoranti, monumenti, spiagge

### Priorità 2 – Social & Community
- **Reddit** – subreddit legati a Cagliari e America's Cup
- **Social Media** (Facebook, Instagram)

### Priorità 3 – Stampa locale *(fallback)*
- Articoli e commenti di testate locali sarde

---

## 🔧 Pipeline di Lavoro

```
Fase 1: Estrazione Dati
    └── Crawler Selenium (giornaliero, per 30 giorni)

Fase 2: Preprocessing
    └── Pulizia, normalizzazione, tokenizzazione

Fase 3: Analisi NLP
    ├── Topic Modeling (LDA / BERTopic)
    └── Sentiment & Emotion Analysis

Fase 4: Analisi Comparativa
    ├── Confronto con baseline 2024
    ├── Confronto con Olbia (città di controllo)
    └── Analisi volumetrica recensioni
```

---

## 📂 Struttura del Repository

```
project/
│
├── scraper/                  # Script di estrazione dati
│   ├── booking_scraper.py
│   ├── airbnb_scraper.py
│   ├── tripadvisor_scraper.py
│   ├── googlemaps_scraper.py
│   └── utils/
│       └── selenium_helpers.py
│
├── data/
│   ├── raw/                  # Dati grezzi estratti
│   ├── processed/            # Dati preprocessati
│   └── baselines/            # Dati anno precedente (2024)
│
├── analysis/
│   ├── preprocessing.py      # Pulizia e normalizzazione testo
│   ├── topic_modeling.py     # LDA / BERTopic
│   ├── sentiment_analysis.py # Sentiment & Emotion Analysis
│   └── comparative.py        # Analisi comparativa finale
│
├── notebooks/                # Jupyter Notebooks esplorativi
│   ├── 01_eda.ipynb
│   ├── 02_topic_modeling.ipynb
│   └── 03_sentiment.ipynb
│
├── results/                  # Output grafici e report
│
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
bertopic
scikit-learn
matplotlib
seaborn
```

---

## 🚀 Utilizzo

### 1. Eseguire lo scraper (consigliato: cron job giornaliero)

```bash
python scraper/booking_scraper.py --city cagliari --start 2025-04-01 --end 2025-06-15
python scraper/booking_scraper.py --city olbia --start 2025-04-01 --end 2025-06-15
```

> ⚠️ **Nota**: Gli scraper usano `time.sleep()` tra le richieste per evitare blocchi. Se il sito inizia a bloccare le richieste, aumentare l'intervallo di pausa nei parametri dello script.

### 2. Preprocessing

```bash
python analysis/preprocessing.py --input data/raw/ --output data/processed/
```

### 3. Analisi NLP

```bash
python analysis/topic_modeling.py
python analysis/sentiment_analysis.py
```

### 4. Analisi comparativa

```bash
python analysis/comparative.py
```

---

## 📊 Output Attesi

- **Distribuzione dei topic** per fonte e finestra temporale
- **Sentiment score medio** per topic (positivo / negativo / neutro)
- **Emozioni prevalenti** per periodo (gioia, sorpresa, frustrazione, ecc.)
- **Variazione volumetrica** delle recensioni rispetto al 2024
- **Confronto Cagliari vs Olbia** per stesso periodo

## 📝 Deliverable Finali

- **Report tecnico** – documento scritto con metodologia, risultati e conclusioni
- **Presentazione PowerPoint** – sintesi visiva dei risultati per la presentazione d'esame

---

## 🧪 Analisi Aggiuntive (Opzionali)

- **PCA** – riduzione dimensionale per visualizzare cluster di topic/sentimenti
- **SVM** – classificazione supervisionata del sentiment
- Integrazione con ~4000 interviste sul campo (pre/post evento) per validazione

---

## 👤 Autore

Progetto universitario Matteo Cambarau– Esame di Web Analytics e Analisi Testuale 

Supervisor: *Professor Marco Ortu*  
Anno Accademico: 2025/2026

---

## 📄 Licenza

Questo progetto è sviluppato a scopo accademico. I dati estratti sono utilizzati esclusivamente per ricerca e non vengono redistribuiti.
