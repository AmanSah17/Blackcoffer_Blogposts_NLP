# Blackcoffer Blog Posts NLP Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, fully object-oriented NLP text analysis pipeline that:

1. **Scrapes** article text from Blackcoffer Insights blog URLs
2. **Analyses** each article across 13 linguistic / sentiment variables
3. **Exports** structured results to Excel — matching the exact output schema specified in the assignment

---

## Project Structure

```
.
├── 20211030 Test Assignment/
│   ├── Input.xlsx                  # 82 URLs with URL_ID mappings
│   ├── Output Data Structure.xlsx  # Reference schema (column order)
│   ├── Final_NLP_Output.xlsx       # ← Generated output (13 metrics × 82 articles)
│   ├── MasterDictionary/
│   │   ├── positive-words.txt
│   │   └── negative-words.txt
│   ├── StopWords/                  # 7 domain-specific stop-word files
│   │   ├── StopWords_Auditor.txt
│   │   ├── StopWords_Currencies.txt
│   │   ├── StopWords_DatesandNumbers.txt
│   │   ├── StopWords_Generic.txt
│   │   ├── StopWords_GenericLong.txt
│   │   ├── StopWords_Geographic.txt
│   │   └── StopWords_Names.txt
│   └── Scraped_Articles/           # 82 × .txt files (one per URL_ID)
└── Notebooks/
    ├── scrape_blackcoffer.py       # Step 1 — Web scraper
    ├── nlp_pipeline.py             # Step 2 — Full NLP analysis pipeline (PRIMARY)
    └── 00_data_loading.ipynb       # Exploratory notebook
```

---

## Computed Variables (13 Metrics)

| # | Variable | Formula / Method |
|---|---|---|
| 1 | POSITIVE SCORE | Count of words matching Positive Dictionary |
| 2 | NEGATIVE SCORE | Count of words matching Negative Dictionary (stored positive) |
| 3 | POLARITY SCORE | `(P − N) / ((P + N) + 0.000001)` — range: `[−1, +1]` |
| 4 | SUBJECTIVITY SCORE | `(P + N) / (Total Cleaned Words + 0.000001)` — range: `[0, +1]` |
| 5 | AVG SENTENCE LENGTH | Total words / Total sentences |
| 6 | PERCENTAGE OF COMPLEX WORDS | Complex words / Total words |
| 7 | FOG INDEX | `0.4 × (Avg Sentence Length + % Complex Words)` |
| 8 | AVG NUMBER OF WORDS PER SENTENCE | Total words / Total sentences |
| 9 | COMPLEX WORD COUNT | Words with > 2 syllables |
| 10 | WORD COUNT | Cleaned words (stop-words + punctuation removed) |
| 11 | SYLLABLE PER WORD | Mean vowel-count per word (handles `es`, `ed` exceptions) |
| 12 | PERSONAL PRONOUNS | Regex count of `I, we, my, ours, us` — excludes country `US` |
| 13 | AVG WORD LENGTH | Total character count / Total words |

---

## Architecture

```
NLPPipeline  (top-level driver)
├── PipelineConfig        — All filesystem paths, validated at startup
├── StopWordLoader        — Merges 7 stop-word files into a frozenset
├── SentimentDictLoader   — Builds positive/negative lexicons (stop-word filtered)
├── TextTokenizer         — NLTK Punkt sentence + word tokenisation
├── WordMetricsAnalyzer   — Syllable counting, pronouns, word lengths
├── SentimentAnalyzer     — Four sentiment scores
├── ReadabilityAnalyzer   — Gunning Fog Index family
└── ArticleAnalyzer       — Per-article orchestrator (5 stages)
```

---

## Setup & Usage

### Prerequisites

```bash
pip install pandas openpyxl nltk tqdm trafilatura beautifulsoup4 selenium requests
```

### Step 1 — Scrape Articles (already done)

```bash
python Notebooks/scrape_blackcoffer.py
```

Reads `Input.xlsx`, scrapes each URL, saves one `.txt` per `URL_ID` into
`20211030 Test Assignment/Scraped_Articles/`.

### Step 2 — Run NLP Analysis Pipeline

```bash
python Notebooks/nlp_pipeline.py
```

Outputs `Final_NLP_Output.xlsx` into the `20211030 Test Assignment/` folder.

**Console output includes:**
- Step-by-step tqdm progress bars
- Live per-article ID during processing
- Preview table (first 5 rows)
- Descriptive statistics for all 13 metrics
- Dual logging: stdout + `nlp_pipeline.log`

---

## Methodology

### Text Cleaning
- All 7 custom `StopWords` files merged into a single set (12,749 stop-words)
- Positive/Negative dictionaries built from `MasterDictionary` **after** removing stop-words

### Sentiment Analysis
Custom lexicon-based approach using the provided domain dictionaries — no black-box model.
Fully reproducible, formula-driven, and auditable against the specification.

### Readability
Gunning Fog Index with NLTK Punkt tokeniser for sentence boundaries.

### Syllable Counting
Vowel-counting heuristic with explicit handling of silent `"es"` and `"ed"` endings.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | DataFrame manipulation and Excel I/O |
| `openpyxl` | `.xlsx` read/write engine |
| `nltk` | Punkt sentence/word tokeniser |
| `tqdm` | Progress bars |
| `trafilatura` | Article extraction (scraper) |
| `beautifulsoup4` | HTML parsing fallback (scraper) |
| `selenium` | JS-rendered page fallback (scraper) |
| `requests` | HTTP fetching (scraper) |

---

## Output Sample

```
               URL_ID   POSITIVE  NEGATIVE  POLARITY  SUBJECTIVITY  FOG INDEX  WORD COUNT
TrackerOPS29012026    16        12        0.143     0.119         7.01       235
TrackerOPS29012027     7         3        0.400     0.063        11.39       159
TrackerOPS29012028    12         6        0.333     0.075        10.18       239
```

---

## Author

**Aman Sah** — Data Extraction & NLP Assignment, Blackcoffer
