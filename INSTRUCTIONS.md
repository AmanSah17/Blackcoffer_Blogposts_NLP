# Blackcoffer NLP Pipeline — Instructions

## How to Run

### 1. Set up the environment

```bash
pip install pandas openpyxl nltk tqdm trafilatura beautifulsoup4 selenium requests python-docx
```

### 2. Configure paths

Open `Notebooks/nlp_pipeline.py` and verify the `root` in `PipelineConfig`:

```python
@dataclass
class PipelineConfig:
    root: Path = field(
        default_factory=lambda: Path(r"D:\gemma4\Test_assignment_20211030")
    )
```

Change `root` to match where you cloned this repository.

### 3. Run the scraper (skip if Scraped_Articles/ already populated)

```bash
python Notebooks/scrape_blackcoffer.py
```

### 4. Run the NLP pipeline

```bash
python Notebooks/nlp_pipeline.py
```

Output: `20211030 Test Assignment/Final_NLP_Output.xlsx`

---

## Approach

### Data Extraction
- Used `trafilatura` (primary) + `BeautifulSoup4` (fallback) + `Selenium` (JS fallback)
- Extracts only article title + body; strips nav, footer, ads, scripts
- Saves as `URL_ID.txt` in `Scraped_Articles/`

### NLP Analysis
- 9-class OOP pipeline: `PipelineConfig`, `StopWordLoader`, `SentimentDictLoader`, 
  `TextTokenizer`, `SentimentAnalyzer`, `ReadabilityAnalyzer`, `WordMetricsAnalyzer`,
  `ArticleAnalyzer`, `NLPPipeline`
- Custom lexicon-based sentiment (MasterDictionary + StopWords folder)
- Gunning Fog readability index
- Vowel-count syllable heuristic with `"es"`/`"ed"` exception handling
- Regex pronoun detection excluding country abbreviation `"US"`

### Dependencies
`pandas`, `openpyxl`, `nltk`, `tqdm`, `trafilatura`, `beautifulsoup4`, `selenium`, `requests`
