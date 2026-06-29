# Tourist Impact Analysis of the America's Cup 2026 in Cagliari

Author: Matteo Cambarau

Course: Web Analytics and Text Analysis — DSBAI, University of Cagliari

Academic Year: 2025/2026

Supervisor: Prof. Marco Ortu

## Executive Summary

This report measures the impact of the America's Cup 2026 on tourist satisfaction in Cagliari, analysing 15,509 Booking.com reviews (Cagliari and Olbia as control city, 2025–2026) through sentiment analysis and topic modeling.

Key results:

| Indicator | Result |
| --- | --- |
| Tourist volume during the event | Reviews +81% compared to the same period in 2025 (333 vs 184) |
| Cagliari rating during the event | Drops from 9.20 (pre) to 8.65 (during), with greater variability |
| Sentiment during the event | From +0.79 (2025 baseline) to +0.56 (2026): −29% drop |
| Post-event recovery | Cagliari sentiment rises to +0.76 in the post-event period, higher than both pre-event (+0.70) and the 2025 baseline (+0.67) |
| Comparison with Olbia (control) | Olbia does not show the same drop during the event — a Cagliari-specific signal |
| Recurring themes | Location, cleanliness, breakfast and staff dominate across all windows; noise and traffic complaints emerge during the event |
| Sentiment model selection (gold standard comparison, 150 reviews) | Feel-IT (58% accuracy) and nlptown (96% accuracy) compared — nlptown selected as the final model. The 96% is partly an artefact: both models struggle to identify negative reviews, which are extremely rare in the dataset |

In summary: the America's Cup generated a significant volumetric increase in tourist arrivals but, during the short event period, a perceived drop in satisfaction — sentiment from +0.79 (2025 baseline) to +0.56 (2026 during), −29% — likely linked to overcrowding, noise and traffic. The post-event period records the highest sentiment value for Cagliari in 2026 (+0.76), above both the pre-event and the 2025 baseline. Section 8 discusses the limitations of the sentiment models used, which are essential for correctly interpreting the absolute values reported.

## Table of Contents

- Executive Summary

- Introduction and Motivation

- Sources, Tools and Libraries

- Phase 1 — Dataset and Data Collection

- Phase 2 — Text Preprocessing

- Phase 3 — NLP Analysis

- 5.1 Sentiment Analysis

- 5.2 Topic Modeling

- Phase 4 — Comparative Analysis

- Phase 5 — Descriptive Temporal Analysis

- Empirical In-Domain Validation (Gold Standard)

- Limitations

- Conclusions

- 11. Appendix — Prompt Engineering: use of LLMs as a development tool

- 11.1 Zero-Shot Prompting — Scraper design

- 11.2 Role Prompting + Output Formatting — LDA topic labelling

- 11.3 Chain-of-Thought (CoT) — Sentiment model selection

- 11.4 Few-Shot Prompting — Manual gold standard annotation

- 11.5 Role + CoT + Output Formatting — Interpreting the Cagliari/Olbia divergence

## 1. Introduction and Motivation

The America's Cup 2026 is one of the most prestigious sailing events in the world and, for the first time in its history, Cagliari will host the races. This represents an extraordinary opportunity for Sardinian tourism, but also a challenge: the city must manage an exceptional influx of visitors within a very concentrated timeframe.

The goal of this project is to measure the perceived impact of the event on tourist satisfaction through the analysis of accommodation reviews. The central research question is: do reviews left by guests during the America's Cup period show significant differences compared to the pre-event period and the previous year?

To answer this question, a text analysis pipeline was built that integrates multiple NLP techniques — sentiment analysis, emotion recognition and topic modeling — applied to reviews scraped from Booking.com.

Why Booking.com? Booking.com is the main hotel booking platform in Europe, with a verified review system (only guests who have stayed can review) that guarantees data authenticity. Reviews are structured with a positive and a negative field, which simplifies sentiment analysis.

Why Olbia as a control city? Olbia is a Sardinian city of comparable size, also with a strong tourism vocation linked to the sea and summer events, but not involved in the America's Cup. Including it allows us to distinguish local event-driven trends from general Sardinian seasonal patterns.

Study design summary:

| Dimension | Treatment | Control |
| --- | --- | --- |
| Temporal | 2026 (event year) | 2025 (same period, previous year) |
| Geographic | Cagliari (event host) | Olbia (no event) |

The design crosses these two dimensions — longitudinal comparison (2026 vs 2025) and synchronic comparison (Cagliari vs Olbia) — to isolate, within the limits allowed by observational data, the specific effect of the event from normal summer tourist seasonality.

## 2. Sources, Tools and Libraries

### 2.1 Data Source

| Source | Usage |
| --- | --- |
| Booking.com | Sole data collection platform. Scraping of reviews (positive/negative text, rating, date, guest country, stay type) for Cagliari and Olbia. |

The file analysis/config.py initially included other sources (airbnb, tripadvisor, google), but for reasons of time and data homogeneity, the project focused exclusively on Booking.com, which alone guarantees sufficient volume (15,509 reviews) and a uniform data structure across both cities.

### 2.2 Development Tools

| Tool | Role in the project |
| --- | --- |
| Python 3.9+ | Main language for scraping, preprocessing and analysis |
| Selenium + ChromeDriver | Browser automation for scraping Booking.com |
| Git | Code and results versioning |
| Visual Studio Code | Main editor |
| Claude (Claude Code) | AI assistant used for iterative generation and debugging of analysis scripts (preprocessing, sentiment, topic modeling, validation) and for project documentation (README.md, CLAUDE.md, this report) |

### 2.3 Libraries Used

Libraries are organised by pipeline phase, with the rationale for each choice.

Scraping and data collection

| Library | Rationale |
| --- | --- |
| selenium | Browser automation required to handle dynamically loaded content (JavaScript) on Booking.com |
| beautifulsoup4 | HTML parsing to isolate individual review fields |

Preprocessing and data handling

| Library | Rationale |
| --- | --- |
| pandas | Main data structure for tabular dataset manipulation |
| nltk | Tokenisation and resource downloads |
| spacy | Italian and English text lemmatization |
| langdetect | Automatic language detection for each review, required to dispatch emotion models (Feel-IT emotion for Italian, DistilRoBERTa for English) |
| textblob | Support utility for text processing |
| sentencepiece | Dependency required by the CamemBERT tokenizer (Feel-IT) |

Sentiment, emotions and topic modeling

| Library | Rationale |
| --- | --- |
| transformers | Loading HuggingFace models (nlptown sentiment, Feel-IT emotion, DistilRoBERTa emotion) |
| datasets | Loading public benchmarks used in the validation module (analysis/validation.py) |
| scikit-learn | LDA implementation (LatentDirichletAllocation), TF-IDF/Count vectorisation, and validation metrics (accuracy, classification report, confusion matrix) |
| bertopic | Semantic embedding-based topic modeling |
| sentence-transformers | Backend for multilingual embeddings used by BERTopic |

Visualisation

| Library | Rationale |
| --- | --- |
| matplotlib | Static chart generation (box plots, bar charts, confusion matrix) |
| seaborn | Confusion matrix heatmaps and uniform chart style |
| wordcloud | Word clouds by polarity/window/city in topic modeling |

## 3. Phase 1 — Dataset and Data Collection

### 3.1 Scraping with Selenium

Data collection was carried out through two Python scrapers built with Selenium (booking_scraper_cagliari.py and booking_scraper_olbia.py), which automate browser navigation to extract reviews from Booking.com.

The scrapers also support automatic checkpointing: if execution is interrupted, they resume from the next property without starting over — an important feature given the duration of scraping sessions across hundreds of properties.

The following fields were collected for each accommodation:

| Field | Description |
| --- | --- |
| city | City of the property |
| property | Property name |
| property_stars | Category (stars) |
| review_date | Review date |
| rating | Numeric score (1–10) |
| review_title | Review title |
| text_positive | Positive section text |
| text_negative | Negative section text |
| stay_type | Stay type (couple, family, etc.) |
| nights_stayed | Number of nights |
| reviewer_country | Reviewer's country |

> Note on column schema: the scraper produces column names slightly different from those used internally by the analysis pipeline (city → property_city, source → scrape_source, property → property_name). The design decision was made not to rename the columns or re-scrape, normalising aliases directly in the analysis scripts (comparative.py, sentiment_analysis.py, topic_modeling.py).

### 3.2 Temporal Windows

Temporal windows were defined around the official race dates:

| Window | 2026 period | 2025 baseline period |
| --- | --- | --- |
| pre | Apr 1 – May 20 | Apr 1 – May 20 |
| during | May 21 – May 24 | May 21 – May 24 |
| post | May 25 – Jun 15 | May 25 – Jun 15 |

The 2026 vs 2025 comparison (same period, previous year) allows isolating the event effect from normal seasonality. Window assignment is based on review_date (the review publication date).

### 3.3 Dataset Composition

The final dataset contains 15,509 reviews distributed as follows:

| City | Window | 2026 | 2025 (baseline) |
| --- | --- | --- | --- |
| Cagliari | pre | 5,090 | 4,256 |
| Cagliari | during | 333 | 184 |
| Cagliari | post | 942 | 1,452 |
| Olbia | pre | 2,048 | — |
| Olbia | during | 86 | — |
| Olbia | post | 1,028 | — |

The Olbia 2025 baseline was deliberately not collected: the Olbia comparative analysis serves as a synchronic control (same period, different city) rather than a longitudinal one.

## 4. Phase 2 — Text Preprocessing

Raw review text requires a cleaning phase before analysis. The analysis/preprocessing.py module performs the following steps:

- Text unification: the text_positive and text_negative fields are processed separately and then joined with a has_negative flag indicating the presence of a negative section.

- Language detection: using the langdetect library, each review is classified as Italian or English. This is essential because sentiment models are language-specific.

- Text cleaning: removal of punctuation, special characters, numbers and multiple spaces.

- Lemmatization: words are reduced to their base form (e.g. "camere" → "camera", "dormito" → "dormire") using spaCy, with the it_core_news_sm model for Italian and en_core_web_sm for English.

- Feature engineering: calculation of review length (review_length) and temporal window assignment (temporal_window) based on the review date.

## 5. Phase 3 — NLP Analysis

### 5.1 Sentiment Analysis (nlptown for all languages)

Sentiment analysis is the core of the project. Given the multilingual nature of the dataset (reviews in Italian and English), a language-specialised model approach was adopted.

#### 5.1.1 Model Used

nlptown/bert-base-multilingual-uncased-sentiment is a multilingual BERT (bert-base-multilingual-uncased) fine-tuned on real product reviews in six languages (German, English, Dutch, French, Italian, Spanish). It classifies text into 1–5 stars, mapped internally in the pipeline as follows:

| Predicted stars | Label | Score |
| --- | --- | --- |
| 1 star | negative | −1.0 |
| 2 stars | negative | −0.5 |
| 3 stars | neutral | 0.0 |
| 4 stars | positive | +0.5 |
| 5 stars | positive | +1.0 |

The model was selected following empirical validation (section 8), which compared Feel-IT and nlptown on a gold standard of 150 manually annotated reviews: nlptown showed 96% accuracy (vs 58% for Feel-IT) and scores more consistent with the numeric ratings in the dataset. Its multilingual architecture, trained specifically on reviews, adapts better to the Booking.com hotel review register compared to Feel-IT, which was trained on tweets. It is used for all languages in the dataset (Italian and English).

#### 5.1.2 Sentiment Results

Average sentiment by city (2026):

| City | Average sentiment | Reviews |
| --- | --- | --- |
| Cagliari | +0.713 | 12,347 |
| Olbia | +0.622 | 3,162 |

With nlptown, sentiment scores are distributed in the [−1, +1] range based on the predicted number of stars. The average sentiment score above +0.70 for Cagliari indicates that the majority of reviews receive 4–5 stars from the model, consistent with the dataset's average numeric ratings (above 8/10).

Average sentiment by temporal window:

| Window | Year | Average sentiment | Reviews |
| --- | --- | --- | --- |
| pre | 2026 | +0.697 | 7,138 |
| during | 2026 | +0.631 | 419 |
| post | 2026 | +0.601 | 1,970 |
| baseline_pre | 2025 | +0.743 | 4,256 |
| baseline_during | 2025 | +0.794 | 184 |
| baseline_post | 2025 | +0.672 | 1,452 |

The most relevant finding is the comparison for the during period: in 2025 the average sentiment was +0.794, while in 2026 it drops to +0.631 (−20.5%). Sentiment remains positive in both years, but the significant drop during the event suggests that tourist pressure negatively affected guest satisfaction compared to the previous year.

### 5.2 Topic Modeling (LDA + BERTopic)

After quantifying the tone of reviews with sentiment analysis, the next step is understanding what they talk about — which themes drive positive and negative ratings, and whether these themes change as a function of the event. Topic modeling identifies recurring themes in reviews in an unsupervised way — without defining categories in advance. Two complementary approaches were applied, run separately on the positive and negative text of each review (polarity).

#### 5.2.1 LDA (Latent Dirichlet Allocation)

LDA is a generative probabilistic model that assumes each document is a mixture of latent topics, and each topic is a word distribution. Parameters used:

- 10 topics per polarity (positive/negative), for a total of 20 models (one per city × window combination)

Positive topics extracted (from results/lda_top_words.csv) describe mainly:

| Topic | Keywords | Interpretation |
| --- | --- | --- |
| 0 | comodo, stanza, molto, colazione, posizione, pulito, bagno, ampio | Room comfort |
| 1 | posizione, pulizia, perfetto, ottimo, stazione, vicino, città, accoglienza, ambiente, staff | Location and hospitality |
| 2 | posizione, ottimo, centro, molto | Centrality |
| 5 | posizione, camera, centro, gentilezza, pulizia | Centrality and staff quality |
| 8 | ottimo, posizione, pulito, colazione, gentile | General positive topic |

Negative topics reflect the most common issues:

| Topic | Keywords | Interpretation |
| --- | --- | --- |
| 1 | camera, rumore, rumoroso, scale | Noise |
| 2 | stanza, insonorizzazione, chiudere | Sound insulation |
| 3 | bagno, piccolo, doccia, asciugamano | Bathroom size and amenities |
| 6 | parcheggio, trovare, mancare | Parking difficulties |
| 8 | nulla, niente, perfetto, segnalare, particolare | No complaints (reviews without issues) |

From the results/topics_by_city.csv table, positive topic 8 is the most frequent in both cities (2,597 occurrences in Cagliari, 640 in Olbia), followed by topic 1 (1,176 in Cagliari, 245 in Olbia). On the negative side, topic 8 dominates by occurrences (2,005 in Cagliari, 491 in Olbia), but its keywords — nulla, niente, perfetto, segnalare, particolare — reveal that it captures reviews with no real complaints. Its dominance is a positive indicator. The main real complaint is topic 1 (noise, rumoroso, scala), second in frequency in both cities — consistent with noise as a cross-cutting issue.

Positive topic 8 is dominant across all temporal windows for both Cagliari and Olbia (results/topics_by_window.csv), confirming that perceived property quality (cleanliness, location, staff friendliness) is stable and independent of the event.

![Figure — LDA topic occurrences for Cagliari](results/lda_by_city.png)

*Figure — LDA topic occurrences for Cagliari and Olbia, split by polarity. Topic 8 (black border) dominates across both polarities and cities. On the negative side, T8 captures reviews without real complaints; the main real complaint is T1 (noise).*

#### 5.2.2 BERTopic

BERTopic is a more modern approach that uses semantic embeddings produced by a multilingual transformer (paraphrase-multilingual-MiniLM-L12-v2) to cluster documents into semantically coherent groups, then applies TF-IDF to extract representative words for each cluster. Compared to LDA, BERTopic better captures semantic relationships between words (e.g. "ottimo" and "eccellente" are treated as similar, not as independent words).

From the results (results/bertopic_top_words.csv), topic 0 — posizione, molto, ottimo, camera, colazione, pulito, pulizia, centro, stanza, comodo — is the largely dominant cluster on the positive side (9,160 occurrences in Cagliari), confirming with an independent method the centrality of the "location + cleanliness + breakfast" factor already identified with LDA. The topic -1 (outlier/unassigned) of BERTopic is the most numerous on the negative side (6,072 in Cagliari, 1,194 in Olbia), which is typical of BERTopic on short, highly heterogeneous negative texts where no dominant semantic cluster emerges.

The following figures show the BERTopic cluster distribution for the two polarities, with bubble size proportional to total occurrences (Cagliari + Olbia).

![Figure — Positive topic clusters (BERTopic).](results/bertopic_cluster_positive.png)

*Figure — Positive topic clusters (BERTopic). Topic 0 (location, cleanliness and breakfast) is clearly dominant. Grey bubbles represent outliers or noise.*

![Figure — Negative topic clusters (BERTopic).](results/bertopic_cluster_negative.png)

*Figure — Negative topic clusters (BERTopic). The "no complaints" topic (purple) is the second largest. The main real complaint is noise and parking (topic 0).*

#### 5.2.3 Visualisations

Generated word clouds show the most frequent terms for each city × window × polarity combination (20 images in results/topic_modeling/). Some representative examples follow.

Positive word cloud — Cagliari, pre-event period 2026:

![Positive word cloud Cagliari pre](results/topic_modeling/wordcloud_cagliari_pre_positive.png)

Figure 1. Most frequent terms in positive Cagliari reviews during the pre-event period (n = 4,937 reviews). Take-away: location, cleanliness and breakfast are the themes driving satisfaction even before the event.

Negative word cloud — Cagliari, during-event period 2026:

![Negative word cloud Cagliari during](results/topic_modeling/wordcloud_cagliari_during_negative.png)

Figure 2. Most frequent terms in negative Cagliari reviews during the event (n = 137 reviews). Take-away: noise and traffic references appear with greater frequency, typical complaints during a concentrated tourist surge.

Positive word cloud — Olbia, pre-event period 2026:

![Positive word cloud Olbia pre](results/topic_modeling/wordcloud_olbia_pre_positive.png)

Figure 3. Most frequent terms in positive Olbia reviews during the pre-event period (n = 1,976 reviews), used as a comparison reference. Take-away: positive themes are very similar to Cagliari's, confirming that location and cleanliness are general satisfaction drivers, not event-specific.

During the event period in Cagliari, noise and traffic-related terms emerge in the negative section, absent or less frequent in the baseline period. This is consistent with the extraordinary influx of visitors that characterises an event of this scale.

## 6. Phase 4 — Comparative Analysis

The analysis/comparative.py module aggregates sentiment and topic modeling results to answer the main research questions.

### 6.1 Volumetric Analysis

The number of reviews is a proxy for tourist volume. The 2026 vs 2025 comparison shows:

| Window | 2026 | 2025 | Change |
| --- | --- | --- | --- |
| pre | 5,090 | 4,256 | +19.6% |
| during | 333 | 184 | +81.0% |
| post | 942 | 1,452 | −35.1% |

The increase in reviews during the event period (+81%) clearly reflects the arrival surge linked to the event. The decline in the post period (−35.1%) is likely due to scraping taking place close to the end of the post window (June 15, 2026), and not all reviews from recent stays had yet been published at the time of collection.

![Figure: Review volume comparison between 2025 baseline](results/volumetric_analysis.png)

*Figure: Review volume comparison between the 2025 baseline (grey) and America's Cup 2026 (black) across the three temporal windows. The +81% peak in the during period highlights the direct event impact on tourist arrivals. The −35.1% drop in the post period should be read with caution: scraping occurred close to the window closing date (June 15, 2026), and a portion of reviews from recent stays was not yet available at the time of collection.*

### 6.2 Comparative Sentiment — Cagliari vs Olbia

Pre-event period (2026):

| City | Average sentiment | % Positive | % Negative | % Neutral |
| --- | --- | --- | --- | --- |
| Cagliari | +0.702 | 89.9% | 1.6% | 8.5% |
| Olbia | +0.683 | 88.3% | 3.3% | 8.4% |

During-event period (2026):

| City | Average sentiment | % Positive | % Negative | % Neutral |
| --- | --- | --- | --- | --- |
| Cagliari | +0.563 | 73.3% | 0.3% | 26.4% |
| Olbia | +0.895 | 95.4% | 1.2% | 3.5% |

Post-event period (2026):

| City | Average sentiment | % Positive | % Negative | % Neutral |
| --- | --- | --- | --- | --- |
| Cagliari | +0.736 | 90.7% | 0.4% | 8.9% |
| Olbia | +0.477 | 75.2% | 8.0% | 16.8% |

The most interesting finding is the divergence between Cagliari and Olbia during the event: Olbia's sentiment rises to +0.895 (the highest value across all periods and cities), while Cagliari drops to +0.563, the lowest value of 2026 for this city. The sharp increase in the "neutral" share in Cagliari (26.4% vs 8.5% in pre) indicates that many reviews land on 3 stars — a signal of reduced satisfaction consistent with overcrowding pressure.

![Figure: Evolution of average sentiment score across the three temporal windows](results/sentiment_score_comparison.png)

*Figure: Evolution of average sentiment score across the three temporal windows. The two cities start at similar values in the pre-event period and diverge sharply during the America's Cup: Cagliari hits its minimum (+0.563) while Olbia reaches its absolute maximum (+0.895). The situation reverses in the post-event period.*

![Figure: Review composition (positive / neutral / negative)](results/sentiment_distribution.png)

*Figure: Review composition (positive / neutral / negative) by city and temporal window. The "neutral" share of Cagliari during the event (26.4%) has almost tripled compared to pre-event (8.5%), a signal of reduced satisfaction consistent with overcrowding pressure.*

![Figure: Sentiment heatmap (green–red scale).](results/sentiment_heatmap.png)

*Figure: Sentiment heatmap (green–red scale). The Olbia/During cell is the greenest (+0.895) and Cagliari/During the reddest (+0.563), making the synchronic divergence between the two cities immediately visible.*

![Figure: Small multiples by temporal window.](results/sentiment_small_multiples.png)

*Figure: Small multiples by temporal window. Each panel directly compares Cagliari (blue) and Olbia (orange), showing parity in the pre period, inversion during the event, and the new reversal in the post.*

### 6.3 Dominant Topics by Temporal Window

Positive topic 8 — characterised by terms such as ottimo, posizione, pulito, colazione, gentile — is dominant across all temporal windows for both cities, confirming that perceived property quality is stable and independent of the event. On the negative side, topic 8 (no-complaint reviews) dominates by volume, but the most recurring real complaint remains topic 1 (noise), present across all temporal windows.

However, during the America's Cup period, topic 0 (camera, colazione, posizione) gains weight at the expense of topic 2 (centro, disponibile, struttura), suggesting a shift of attention towards basic comforts — consistent with a more crowded and less relaxed stay experience.

![Figure: Heatmap of the percentage distribution of negative topics](results/topics_neg_heatmap.png)

*Figure: Heatmap of the percentage distribution of negative LDA topics across the three temporal windows. Topic 8 (no-complaint reviews) is excluded to surface the real signal. The intense red on T0 (room/breakfast/location, 24.4%) and T4 (price/value-for-money, 21.2%) during the event clearly highlights the discomfort perceived by tourists during the America's Cup period, in stark contrast with the pre and post values.*

## 7. Phase 5 — Descriptive Temporal Analysis

While the previous sections analysed sentiment and topics in aggregate form, this section adopts a purely descriptive and temporal perspective, comparing rating and sentiment distributions across the pre/during/post windows and, for Cagliari, between 2026 and the 2025 baseline.

### 7.1 Rating by Temporal Window (2026)

The following box plot shows the rating distribution across the three temporal windows, separated by city:

![Box plot rating by temporal window](results/exploratory/temporal_rating_boxplot.png)

Figure 4. Distribution of rating (1–10) by temporal window (pre/during/post) and city, with mean (◆) and sample sizes indicated. Take-away: in Cagliari the average rating drops during the event (8.65 vs 9.20 in pre) and shows greater variability, while in Olbia the pattern is reversed.

Main observations (from results/exploratory/temporal_summary.csv):

| City | Window | n | Average rating | Std | Average sentiment |
| --- | --- | --- | --- | --- | --- |
| Cagliari | pre | 93,878 | 9.204 | 0.888 | +0.699 |
| Cagliari | during | 6,045 | 8.650 | 1.134 | +0.545 |
| Cagliari | post | 10,064 | 9.156 | 0.933 | +0.762 |
| Olbia | pre | 34,018 | 8.842 | 1.063 | +0.684 |
| Olbia | during | 1,188 | 9.643 | 0.493 | +0.992 |
| Olbia | post | 16,654 | 8.695 | 1.145 | +0.437 |

> Note: the n values in this table are rows of the merged dataset expanded by the topic modeling join (multiple topics per review), not unique reviews — they should be read as relative weights, not review counts.

- In Cagliari the average rating drops during the event (8.65) compared to pre (9.20) and recovers in the post (9.16); the same pattern emerges in sentiment (+0.699 → +0.545 → +0.762)

- The rating standard deviation increases during the event (1.134 vs 0.888 in pre), indicating greater variability in experiences

- Olbia shows the opposite pattern: both rating (9.643) and sentiment (+0.992) during the event are the highest values of the three periods, but the sample is very small (n=1,188 rows, corresponding to 86 unique reviews)

### 7.2 Sentiment by Temporal Window (2026)

![Box plot sentiment by temporal window](results/exploratory/temporal_sentiment_boxplot.png)

Figure 5. Distribution of sentiment score by temporal window and city. Take-away: in Cagliari sentiment drops during the event (+0.545 vs +0.699 in pre) and recovers in the post-event period (+0.762), while Olbia shows its maximum value precisely during the event (+0.992).

The sentiment score confirms the rating pattern: in Cagliari the lowest 2026 value is recorded in the during period (+0.545), while the post-event (+0.762) exceeds the pre (+0.699) — a positive "rebound" effect after the pressure of the racing period.

### 7.3 2026 vs 2025 Baseline Comparison — Cagliari

The following chart is the most directly relevant result for the research question: it compares average rating and sentiment across the same temporal windows between 2026 (America's Cup year) and 2025 (reference year):

![Cagliari 2026 vs 2025 comparison](results/exploratory/temporal_cagliari_2026_vs_2025.png)

Figure 6. Direct comparison, for Cagliari, between average rating and sentiment in 2026 (event year) and the same period in 2025 (baseline), with sample sizes indicated. Take-away: the during period shows the sharpest drop compared to the baseline (sentiment from +0.79 to +0.56, −29%), while the 2026 post-event period exceeds the 2025 baseline.

Interpretation:

- Rating: in the during period, the 2026 average rating is lower than 2025 (8.65 vs ~8.9 in 2025). In the post period, 2026 records slightly higher values than 2025.

- Sentiment: the sharpest drop is in the during period, where 2025 sentiment was +0.794 while 2026 falls to +0.563 (−29%). In 2025, late May was a quiet period; in 2026 the event's influx introduced sources of dissatisfaction, without pushing sentiment into negative territory.

The 2026 post-event period shows sentiment of +0.736, above the 2025 baseline (+0.672): guests who stay after the event tend to leave more positive reviews, probably because tourist pressure has eased while enthusiasm for the city remains high.

## 8. Empirical In-Domain Validation (Gold Standard)

To quantify the reliability of sentiment results, an empirical in-domain validation was conducted on a manually annotated review sample. Two models were compared: Feel-IT (MilaNLProc/feel-it-italian-sentiment) and nlptown/bert-base-multilingual-uncased-sentiment. The validation results guided the decision to adopt nlptown as the main pipeline model (section 5.1), replacing Feel-IT for Italian sentiment and VADER for English — nlptown now handles all languages with a single model.

### 8.1 Gold Standard Construction

The script analysis/generate_gold_standard.py randomly selected 150 Italian-language reviews from the Cagliari corpus, sampling 30 reviews per temporal window (pre, during, post, baseline_pre, baseline_during) to ensure a balanced distribution. Rows were shuffled in random order (random_state=42) to avoid position bias during the annotation phase.

For each review, the following fields were included: sequential id, temporal window, date, numeric rating, text_positive and text_negative. Manual annotation assigned a positive or negative label based on reading the text and the rating, following this criterion:

- Rating ≥ 8 with dominant positive text → positive

- Rating 6 with explicit and strong negative text → negative

- Rating 7 with significant complaints in text_negative → negative

Gold standard distribution (n=150):

| Label | Count | % |
| --- | --- | --- |
| positive | 147 | 98.0% |
| negative | 3 | 2.0% |

The imbalance reflects the actual corpus distribution: Booking.com reviews are structurally oriented towards positive evaluation, with typically high ratings.

### 8.2 Inference and Results: Feel-IT vs nlptown

The script analysis/empirical_validation.py applied both models to the 150 gold standard texts.

#### 8.2.1 Feel-IT (MilaNLProc/feel-it-italian-sentiment)

Overall metrics:

| Metric | Value |
| --- | --- |
| Accuracy | 58.0% |
| Macro F1 | 0.395 |
| Weighted F1 | 0.716 |

Precision / Recall / F1 by class:

| Class | Precision | Recall | F1 | n |
| --- | --- | --- | --- | --- |
| negative | 0.031 | 0.667 | 0.060 | 3 |
| positive | 0.988 | 0.578 | 0.730 | 147 |

Accuracy by temporal window:

| Window | n | Accuracy |
| --- | --- | --- |
| baseline_during | 30 | 50.0% |
| baseline_pre | 30 | 56.7% |
| during | 30 | 66.7% |
| post | 30 | 60.0% |
| pre | 30 | 56.7% |

![Feel-IT confusion matrix](validation/outputs/confusion_matrix_feelit.png)

Figure 7. Feel-IT confusion matrix on the gold standard of 150 manually annotated reviews. Take-away: the model correctly identifies 2 of the 3 real negative cases (recall 66.7%), but at the cost of 62 false positives on the positive class (precision 3.1%).

![Feel-IT accuracy by temporal window](validation/outputs/accuracy_per_finestra.png)

Figure 8. Feel-IT accuracy computed separately for each of the 5 temporal windows of the gold standard (30 reviews each), with the overall average indicated by the dashed line. Take-away: accuracy oscillates between 50% and 66.7% across all windows, indicating a systematic model error independent of the period considered.

#### 8.2.2 nlptown (nlptown/bert-base-multilingual-uncased-sentiment)

nlptown/bert-base-multilingual-uncased-sentiment is a multilingual BERT (bert-base-multilingual-uncased) fine-tuned on product reviews in six languages (including Italian), returning a 5-star classification (1–5). To make it comparable with the binary gold standard labelling, the output was mapped via _MAPPA_STELLE:

| Predicted stars | Mapped label |
| --- | --- |
| 1 star, 2 stars | negative |
| 3 stars, 4 stars, 5 stars | positive |

This mapping is deliberately conservative: even a "medium" judgment (3 stars) is counted as positive, which favours the gold standard's majority class.

Overall metrics:

| Metric | Value |
| --- | --- |
| Accuracy | 96.0% |
| Macro F1 | 0.490 |
| Weighted F1 | 0.960 |

Precision / Recall / F1 by class:

| Class | Precision | Recall | F1 | n |
| --- | --- | --- | --- | --- |
| negative | 0.000 | 0.000 | 0.000 | 3 |
| positive | 0.980 | 0.980 | 0.980 | 147 |

![nlptown confusion matrix](validation/outputs/confusion_matrix_nlptown.png)

Figure 9. nlptown confusion matrix on the same gold standard. Take-away: nlptown correctly identifies none of the 3 real negative cases (recall 0%): the 96% accuracy is therefore almost entirely due to the weight of the positive majority class.

### 8.3 Comparison Between the Two Models: Why Accuracy Alone Is Misleading

Comparison table (validation/outputs/confronto_modelli.csv):

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- |
| Feel-IT (MilaNLProc) | 58.0% | 0.395 | 0.716 |
| nlptown (bert-multilingual) | 96.0% | 0.490 | 0.960 |

At first glance, nlptown appears clearly superior: 96% accuracy vs 58% for Feel-IT. But the confusion matrix reveals a more nuanced picture. Of the 3 genuinely negative reviews, nlptown correctly identifies none (precision = recall = 0 on the negative class): all 3 are classified as positive. The 96% accuracy is therefore almost entirely an artefact of the majority class — nlptown behaves, on this gold standard, similarly to a naive classifier that always predicts "positive" (which would achieve 147/150 = 98.0% accuracy).

Feel-IT, conversely, correctly identifies 2 of the 3 negative reviews (recall = 0.667), but at the cost of a high number of false positives on the negative class — that is, 62 of the 147 genuinely positive reviews are incorrectly labelled as negative (precision = 0.031).

In other words: neither model handles the minority negative class adequately, although the failures are of opposite sign: nlptown systematically ignores negative cases (recall = 0), while Feel-IT tends to over-signal them, producing a high number of false positives. Macro F1 (which weights both classes equally, regardless of their size) makes this trade-off more visible: 0.395 for Feel-IT vs 0.490 for nlptown — a much smaller gap than the accuracy jump (58% vs 96%), and neither value is high in absolute terms.

Methodological lesson: with a heavily imbalanced gold standard (147 positives out of 150, 98%), accuracy rewards any model that tends to predict the majority class, regardless of its actual ability to distinguish between classes. Macro F1 (or alternatively Precision/Recall per class) is the most informative metric in this context.

Feel-IT's behaviour is explainable by the training domain mismatch: Feel-IT is trained primarily on Italian tweets, characterised by informal, ironic and often aggressive register. Booking.com hotel reviews have a more neutral and descriptive register, with negative sentences often phrased in a moderate way (e.g. "la zona non è sicurissima di notte", "mancanza di parcheggio"). The model interprets these constructs as negative, even when the guest's overall experience was satisfactory. nlptown, trained on real product reviews, has a register closer to Booking.com, but its 5-star mapping — combined with extreme class imbalance in the gold standard — makes it unable to flag the (few) negative cases.

Implications for the project's sentiment analysis (which uses nlptown as the main model for Italian, following this validation):

- Absolute sentiment values with nlptown (e.g. +0.713 average for Cagliari) are more directly interpretable than Feel-IT, since they derive from a linear mapping of the predicted star count — register similar to Booking.com's numeric ratings.

- Relative comparisons (Cagliari during 2026 vs during 2025, Cagliari vs Olbia) remain the most reliable metric: the model's systematic bias towards the positive class applies uniformly across all windows and cities, making group differences more robust than absolute values.

- The direction of observed trends (drop during the event, post-event recovery) is consistent with numeric rating variations, an NLP-independent measure not subject to the same bias.

- The main residual limitation of nlptown is its low sensitivity to the negative class (recall 0% on 3 cases in the gold standard): the model tends to always predict "positive" for hotel reviews, making the absolute count of negative reviews (sections 6.2 and 5.2, negative topics) unreliable. Relative trends remain informative nonetheless.

## 9. Limitations

The results presented should be read in light of the following methodological limitations, which do not invalidate the main conclusions but define the boundaries of interpretation:

- Low sensitivity to the negative class in nlptown: the empirical validation (section 8) showed that nlptown achieves 96% accuracy but recall = 0% on the negative class (0 of the 3 negative cases identified). The model systematically tends to predict "positive", limiting its usefulness for identifying critical reviews. Relative comparisons between windows/cities remain valid (the bias is systematic), while absolute counts of negative reviews (sections 6.2 and 5.2) should be interpreted with caution.

- Gold standard imbalance: 147 positive examples vs 3 negative (98%/2%) makes metrics for the negative class (Precision 3.1%, n=3) statistically weak — a single misclassification counts for ~33 percentage points on that class.

- Partial post period: scraping occurred close to the end of the post window (June 15, 2026), so a portion of reviews related to post-event stays had not yet been published at the time of collection. The −35.1% decline compared to 2025 in the post period (section 6.1) should be read in light of this effect, not only as a genuine signal of declining arrivals.

- Absence of the 2025 baseline for Olbia: the control city only has 2026 data, which limits longitudinal comparison for Olbia to the sole synchronic comparison with Cagliari (section 6.2), without the ability to verify whether Olbia also experienced year-on-year variations independent of the event.

- Single data source: the analysis is based exclusively on Booking.com. Other platforms (TripAdvisor, Google Reviews, Airbnb), originally planned in config.py, were not integrated; the demographic profile and expectations of Booking.com users may not be representative of all visitors.

## 10. Conclusions

The results of this project show that online reviews are an effective tool for measuring the impact of a major event on the tourist perception of a city.

### Main Findings

- Increased arrivals: the number of reviews during the America's Cup increased by 81% compared to the same period in 2025, confirming the volumetric impact of the event.

- Sentiment worsening during the event: the average rating in Cagliari drops from 9.20 (pre) to 8.65 (during), and sentiment moves from +0.699 to +0.545. The comparison with the 2025 baseline is even more marked: +0.794 in 2025 vs +0.563 in 2026 (−29%).

- Post-event recovery: after the races, sentiment in Cagliari rises to +0.762, the highest value of 2026 for this city and above the 2025 baseline (+0.672), suggesting a positive "long-tail" effect of the event on destination image.

- Olbia as a control: the control city does not show Cagliari's worsening during the event (in fact, it shows the highest average rating of the three periods), confirming that the detected signal is specific to the host city.

- Stable topics: recurring themes (location, cleanliness, staff, breakfast) do not change substantially across temporal windows — confirmed by both LDA and BERTopic — but during the event, noise and traffic complaint signals emerge.

### Technology Stack — Summary

| Component | Technology |
| --- | --- |
| Scraping | Python, Selenium |
| Preprocessing | spaCy, NLTK, langdetect |
| Italian Sentiment | nlptown (bert-base-multilingual-uncased-sentiment) |
| English Sentiment | nlptown (same model as Italian) |
| Italian Emotions | Feel-IT emotion |
| English Emotions | DistilRoBERTa (Hartmann et al.) |
| Topic modeling | LDA (scikit-learn), BERTopic |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Validation | scikit-learn (metrics), in-domain gold standard (150 reviews) — Feel-IT vs nlptown; nlptown adopted following results; HuggingFace benchmarks (datasets) for further steps |
| Visualisations | matplotlib, seaborn, wordcloud, pyLDAvis |

Report generated from the analysis pipeline results available in the project repository.

## 11. Appendix — Prompt Engineering: Use of LLMs as a Development Tool

During project development, large language models (LLMs) — in particular Claude (Anthropic) — were used as support tools in several phases: code design, results interpretation, NLP model selection and report writing. This appendix documents the main prompts used, explaining the prompt engineering techniques adopted to obtain accurate and contextualised outputs.

The techniques illustrated are: zero-shot prompting, few-shot prompting, chain-of-thought (CoT), role prompting and output formatting. Each prompt is accompanied by a methodological note justifying its structure.

────────────────────────────────────────────────────────────

### 11.1 Zero-Shot Prompting — Scraper Design

Context: initial project phase, need to build a web scraper for Booking.com with Selenium.

Technique: zero-shot — the model receives only the task description, without examples. Suitable for well-defined tasks where the model already has sufficient domain knowledge.

Write a Python script with Selenium that downloads hotel reviews for Cagliari
from Booking.com. For each review extract the following fields:
- hotel name and number of stars
- review date
- numeric rating (1–10 scale)
- positive and negative text separately
- reviewer's country, stay type and number of nights

Requirements:
- handle lazy loading and pagination (multiple result pages)
- save output to a CSV file with header
- add retry on timeout
- use explicit waits (WebDriverWait) instead of time.sleep()

Why it works: the request is structured with bullet points and explicit technical requirements. Zero-shot works well here because Selenium and Booking.com are in the model's training data, and the clear field specification limits output ambiguity.

────────────────────────────────────────────────────────────

### 11.2 Role Prompting + Output Formatting — LDA Topic Labelling

Context: after running topic_modeling.py, the 10 LDA topics are represented as keyword lists. A readable semantic label needs to be assigned to each one.

Technique: role prompting (assigning an expert role to the model) combined with output formatting (specifying the exact output format). Reduces vagueness and produces output directly usable in code.

You are a computational linguist expert in tourism topic modeling.
I provide the 10 most representative words of each topic extracted via LDA
from Italian hotel reviews (corpus: ~15,000 reviews, Booking.com).

For each topic assign a short semantic label (maximum 4 words).
Return the results EXACTLY in the following Python format, with no additional text:

TOPIC_LABELS_POSITIVE = {
    0: "label",
    1: "label",
    ...
}

Positive topics (terms ordered by weight):
- Topic 0: ottimo, posizione, pulito, colazione, gentile, personale, camera, comodo, bello, vicino
- Topic 1: centro, struttura, disponibile, servizio, hotel, buono, soggiorno, piacevole, utile
[...]

Why it works: the role contextualises the domain and improves the semantic quality of the labels. The format constraint (Python dictionary) produces output that can be pasted directly into code without post-processing.

────────────────────────────────────────────────────────────

### 11.3 Chain-of-Thought (CoT) — Sentiment Model Selection

Context: after empirical validation (section 8), need to choose between Feel-IT and nlptown as the main pipeline model.

Technique: chain-of-thought — asking the model to reason step by step before formulating a recommendation. Produces more articulated reasoning less prone to oversimplification.

I need to choose the main sentiment analysis model for a corpus of 15,509
hotel reviews in Italian and English (Booking.com, Cagliari and Olbia, 2025–2026).

I compared two models on a manually annotated gold standard (150 reviews):

Feel-IT (MilaNLProc/feel-it-italian-sentiment):
  Accuracy: 58.0% | Macro F1: 0.395
  Negative class: Precision 0.031, Recall 0.667, F1 0.060
  Positive class: Precision 0.988, Recall 0.578, F1 0.730

nlptown (bert-base-multilingual-uncased-sentiment, 1-2 stars=neg, 3-5=pos):
  Accuracy: 96.0% | Macro F1: 0.490
  Negative class: Precision 0.000, Recall 0.000, F1 0.000
  Positive class: Precision 0.980, Recall 0.980, F1 0.980

Reason step by step considering:
1. The training domain of each model vs the target domain
2. The implications of recall=0 on the negative class vs precision=0.031
3. The reliability of relative comparisons (windows, cities) with each model
4. The fact that the gold standard has only 3 negative examples out of 150

Finally, indicate which model you would choose and why, in 3–4 lines
suitable for a university technical report.

Why it works: listing the reasoning steps forces the model to consider all dimensions of the problem. Without CoT, it tends to choose the model with higher accuracy (96%) without examining the class imbalance paradox.

────────────────────────────────────────────────────────────

### 11.4 Few-Shot Prompting — Manual Gold Standard Annotation

Context: gold standard construction (section 8.1). Need to define a consistent labelling criterion for the 150 reviews to be manually annotated.

Technique: few-shot — providing 3–5 labelled examples before the request. Particularly effective for classification tasks where the category definition is nuanced (e.g. distinguishing a negative review from a mixed one).

I need to label hotel reviews as 'positive' or 'negative' to build
a gold standard for NLP model validation. Follow these examples:

EXAMPLES:
---
Positive text: "Excellent location in the historic centre, very friendly staff.
Generous breakfast. Clean and comfortable room."
Negative text: ""
Rating: 9/10 → Label: positive
Rationale: high rating, text with no real complaints.

---
Positive text: ""
Negative text: "Tiny dark room, run-down bathroom, street noise all night.
We won't be back."
Rating: 4/10 → Label: negative
Rationale: low rating, dominant and explicit negative text.

---
Positive text: "Convenient location near the port. Courteous staff."
Negative text: "The room was small. No parking."
Rating: 7/10 → Label: positive
Rationale: medium-high rating, complaints are minor and don't invalidate the experience.

---
Now label the following reviews applying the same criterion.
For each review indicate: label and brief rationale (max 10 words).
[list of reviews to label]

Why it works: the three examples cover prototypical cases (positive, negative, borderline). The short rationale requirement makes the reasoning verifiable and consistent across annotations.

────────────────────────────────────────────────────────────

### 11.5 Role + CoT + Output Formatting — Interpreting the Cagliari/Olbia Divergence

Context: section 6.2 of the report. Data show a sharp divergence during the America's Cup (+0.895 Olbia vs +0.563 Cagliari). Need to interpret the pattern rigorously.

Technique: combination of role prompting, chain-of-thought and output formatting. The combination is useful when the task requires domain expertise, argumentative rigour and a specific format for report insertion.

You are a tourism economics researcher writing a paper on the impact
of major sporting events on the perceived quality of accommodation.

Analyse the following sentiment data extracted from Booking.com reviews (2026):

Cagliari (America's Cup host city):
  Pre-event (Apr 1 – May 20):   average sentiment +0.702, average rating 9.20
  During (May 21–24):           average sentiment +0.563, average rating 8.65
  Post-event (May 25 – Jun 15): average sentiment +0.736, average rating 9.16

Olbia (control city, same region, no event):
  Pre-event:   average sentiment +0.683, average rating 8.84
  During:      average sentiment +0.895, average rating 9.64
  Post-event:  average sentiment +0.477, average rating 8.70

Reason step by step on:
1. What explains the divergence between Cagliari and Olbia during the event
2. Why Olbia records its highest value precisely during the America's Cup
3. What the post-event decline of Olbia compared to Cagliari's recovery indicates
4. How to read the results in light of nlptown's bias towards the positive class

Required output: 3 bullet points in English (maximum 2 lines each),
academic tone, suitable for direct insertion in a university report.

Why it works: the combination is the most powerful for complex interpretive tasks. The role ensures the academic register; CoT prevents superficial interpretations; the format constraint makes the output immediately usable in the report.

────────────────────────────────────────────────────────────

### Methodological Note on the Use of LLMs in the Project

The use of LLMs as development assistants accelerated operational phases (code writing, output reformatting, internal consistency checks on results) and supported the interpretive phase by providing a second, argued perspective.

All quantitative results in the report come exclusively from the execution of the Python pipeline (analysis/): the NLP models used for sentiment and topic modeling are nlptown, Feel-IT and BERTopic, not generative LLMs. LLMs were used as a working tool, not as an analytical component.
