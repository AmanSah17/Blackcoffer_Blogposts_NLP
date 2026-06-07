"""
nlp_pipeline.py
===============
sNLP Text Analysis Pipeline
============================================

Author      : Aman Sah , Email : amansah1717@gmail.com
Description : A fully object-oriented, modular, and documented NLP pipeline
              that reads scraped article text files, computes 13 linguistic
              and sentiment variables as specified in ``Text Analysis.docx``,
              and persists the results in an Excel workbook.

Algorithm Overview (per Text Analysis.docx)
--------------------------------------------
1.  Load & merge all custom Stop-Word lists (7 domain-specific files).
2.  Build Positive / Negative sentiment dictionaries from the Master
    Dictionary after stripping stop-words.
3.  For every article text file, run a five-stage analysis:
        Stage A – Tokenisation   : sentences and words via NLTK.
        Stage B – Sentiment      : Positive Score, Negative Score,
                                   Polarity Score, Subjectivity Score.
        Stage C – Readability    : Avg Sentence Length, % Complex Words,
                                   Fog Index, Avg Words/Sentence.
        Stage D – Word Metrics   : Complex Word Count, Word Count,
                                   Syllable/Word, Avg Word Length.
        Stage E – Pronouns       : Personal Pronoun count via regex.
4.  Assemble a pandas DataFrame and write it to Excel.

Classes
-------
- PipelineConfig      : Central dataclass holding all filesystem paths.
- StopWordLoader      : Reads and merges multi-file stop-word lists.
- SentimentDictLoader : Builds positive / negative word sets.
- TextTokenizer       : Cleans raw text, tokenises into sentences & words.
- SentimentAnalyzer   : Computes the four sentiment scores.
- ReadabilityAnalyzer : Computes Gunning Fog and derived readability metrics.
- WordMetricsAnalyzer : Computes syllable, pronoun, and length statistics.
- ArticleAnalyzer     : Orchestrates all five stages for one article.
- NLPPipeline         : Top-level driver; reads input, iterates articles,
                        writes output. Entry point for the entire workflow.

Usage
-----
    python nlp_pipeline.py

Dependencies
------------
    pip install pandas openpyxl nltk tqdm
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import nltk
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# NLTK corpus bootstrap
#   We download silently so the script is self-contained; downloads are
#   skipped automatically when the data is already present on disk.
# ---------------------------------------------------------------------------
for _corpus in ("punkt", "punkt_tab"):
    nltk.download(_corpus, quiet=True)

# ---------------------------------------------------------------------------
# Module-level logger
#   Every class in this module emits structured log messages so operators
#   can diagnose failures without touching the source code.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s » %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("nlp_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# ①  PipelineConfig
#    ──────────────
#    Central configuration dataclass.  All filesystem paths are resolved
#    here once so that every downstream class can simply reference
#    ``config.<attribute>`` instead of re-building paths ad hoc.
# ===========================================================================
@dataclass
class PipelineConfig:
    """
    Immutable configuration container for the NLP pipeline.

    All path attributes are :class:`pathlib.Path` objects so the pipeline
    is portable across Windows, macOS, and Linux without string manipulation.

    Parameters
    ----------
    root : Path
        The top-level workspace directory that contains the
        "20211030 Test Assignment" sub-folder.

    Attributes
    ----------
    assignment_dir : Path
        Resolved path to the "20211030 Test Assignment" folder.
    input_file : Path
        Path to ``Input.xlsx`` (contains URL_ID → URL mapping).
    scraped_dir : Path
        Directory that holds one ``.txt`` file per URL_ID.
    stopwords_dir : Path
        Directory containing the seven ``StopWords_*.txt`` files.
    master_dict_dir : Path
        Directory containing ``positive-words.txt`` and
        ``negative-words.txt``.
    output_file : Path
        Destination Excel workbook for the analysis results.
    encoding : str
        Fallback encoding used for all plain-text dictionary files.
        ``"latin-1"`` handles the occasional non-ASCII byte found in the
        supplied word lists.
    """

    root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    encoding: str = "latin-1"

    # Derived paths are computed in __post_init__ so callers never need
    # to repeat the same path arithmetic in multiple places.
    assignment_dir: Path = field(init=False)
    input_file: Path = field(init=False)
    scraped_dir: Path = field(init=False)
    stopwords_dir: Path = field(init=False)
    master_dict_dir: Path = field(init=False)
    output_file: Path = field(init=False)

    def __post_init__(self) -> None:
        """Resolve all derived paths from ``self.root``."""
        self.assignment_dir = self.root / "20211030 Test Assignment"
        self.input_file     = self.assignment_dir / "Input.xlsx"
        self.scraped_dir    = self.assignment_dir / "Scraped_Articles"
        self.stopwords_dir  = self.assignment_dir / "StopWords"
        self.master_dict_dir = self.assignment_dir / "MasterDictionary"
        self.output_file    = self.assignment_dir / "Final_NLP_Output.xlsx"
        self._validate()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """
        Assert that every critical directory / file exists on disk.

        Raises
        ------
        FileNotFoundError
            If any required path is absent, a descriptive error is raised
            immediately so the user gets a clear message rather than a
            confusing ``KeyError`` later in the pipeline.
        """
        required: Dict[str, Path] = {
            "Root directory":      self.root,
            "Assignment folder":   self.assignment_dir,
            "Input.xlsx":          self.input_file,
            "Scraped articles dir": self.scraped_dir,
            "StopWords directory": self.stopwords_dir,
            "MasterDictionary dir": self.master_dict_dir,
        }
        for label, path in required.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"[PipelineConfig] {label} not found → {path}\n"
                    "Please verify the 'root' path supplied to PipelineConfig."
                )
        logger.info("PipelineConfig validated — all paths resolved successfully.")


# ===========================================================================
# ②  StopWordLoader
#    ──────────────
#    Reads all seven domain-specific StopWord files and merges them into a
#    single frozenset for O(1) membership testing throughout the pipeline.
# ===========================================================================
class StopWordLoader:
    """
    Aggregate loader for the multi-file custom stop-word corpus.

    The supplied ``StopWords/`` directory contains seven files covering
    auditor names, currencies, dates, generic function words, geographic
    terms, personal names, and a comprehensive generic-long list.  Each
    file may use the pipe-character ``|`` as a comment delimiter (content
    after ``|`` is ignored).

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration object; only ``config.stopwords_dir``
        and ``config.encoding`` are consumed.

    Attributes
    ----------
    stop_words : FrozenSet[str]
        Lower-cased, stripped stop-words across all loaded files.

    Examples
    --------
    >>> cfg = PipelineConfig()
    >>> loader = StopWordLoader(cfg)
    >>> "about" in loader.stop_words
    True
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config: PipelineConfig = config
        self.stop_words: FrozenSet[str] = self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_line(self, raw_line: str) -> Optional[str]:
        """
        Parse a single line from a stop-word file.

        Lines may contain comments after a pipe character.  Leading and
        trailing whitespace is stripped.  Empty lines and comment-only
        lines return ``None``.

        Parameters
        ----------
        raw_line : str
            One raw line read from a stop-word file.

        Returns
        -------
        Optional[str]
            The cleaned, lower-cased token, or ``None`` if the line
            carries no usable word.
        """
        # Strip the comment portion (everything after ``|``)
        token = raw_line.split("|")[0].strip().lower()
        return token if token else None

    def _load_single_file(self, filepath: Path) -> Set[str]:
        """
        Read one stop-word file and return a set of clean tokens.

        Parameters
        ----------
        filepath : Path
            Absolute path to the ``.txt`` stop-word file.

        Returns
        -------
        Set[str]
            All valid lower-cased stop-words extracted from the file.
        """
        words: Set[str] = set()
        try:
            with filepath.open("r", encoding=self._config.encoding) as fh:
                for raw_line in fh:
                    token = self._parse_line(raw_line)
                    if token:
                        words.add(token)
        except OSError as exc:
            # Log but do not abort; partial stop-word data is better than
            # crashing the entire pipeline for a missing ancillary file.
            logger.warning(
                "StopWordLoader: could not read '%s' — %s", filepath.name, exc
            )
        return words

    def _load(self) -> FrozenSet[str]:
        """
        Iterate over every ``.txt`` file in the stop-words directory and
        merge all tokens into a single immutable frozenset.

        Returns
        -------
        FrozenSet[str]
            Combined, de-duplicated set of all stop-words.
        """
        combined: Set[str] = set()
        txt_files: List[Path] = sorted(
            self._config.stopwords_dir.glob("*.txt")
        )
        if not txt_files:
            logger.warning(
                "StopWordLoader: no .txt files found in '%s'.",
                self._config.stopwords_dir,
            )
        for filepath in txt_files:
            file_words = self._load_single_file(filepath)
            combined |= file_words
            logger.debug(
                "  Loaded %d stop-words from '%s'",
                len(file_words),
                filepath.name,
            )
        logger.info(
            "StopWordLoader: %d unique stop-words loaded from %d file(s).",
            len(combined),
            len(txt_files),
        )
        return frozenset(combined)


# ===========================================================================
# ③  SentimentDictLoader
#    ────────────────────
#    Builds Positive and Negative word sets from the Master Dictionary
#    after filtering out any entry already present in the stop-word corpus.
# ===========================================================================
class SentimentDictLoader:
    """
    Builder for the domain-specific positive / negative sentiment lexicons.

    Per the assignment specification (Text Analysis.docx §2):

        *"We add only those words in the dictionary if they are not found
        in the Stop Words Lists."*

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration object.
    stop_words : FrozenSet[str]
        Pre-built stop-word set from :class:`StopWordLoader`.

    Attributes
    ----------
    positive_words : FrozenSet[str]
        Lower-cased positive-sentiment words, stop-word filtered.
    negative_words : FrozenSet[str]
        Lower-cased negative-sentiment words, stop-word filtered.
    """

    def __init__(
        self,
        config: PipelineConfig,
        stop_words: FrozenSet[str],
    ) -> None:
        self._config: PipelineConfig = config
        self._stop_words: FrozenSet[str] = stop_words
        self.positive_words: FrozenSet[str]
        self.negative_words: FrozenSet[str]
        self.positive_words, self.negative_words = self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_word_file(
        self,
        filepath: Path,
        label: str,
    ) -> FrozenSet[str]:
        """
        Read a single-column word-list file and return a stop-word-filtered
        frozenset.

        Parameters
        ----------
        filepath : Path
            Absolute path to the word-list file.
        label : str
            Human-readable label used in log messages (e.g., ``"positive"``).

        Returns
        -------
        FrozenSet[str]
            Filtered word set.
        """
        words: Set[str] = set()
        if not filepath.exists():
            logger.error(
                "SentimentDictLoader: %s-words file not found → %s",
                label,
                filepath,
            )
            return frozenset()

        with filepath.open("r", encoding=self._config.encoding) as fh:
            for raw_line in fh:
                token = raw_line.strip().lower()
                # Skip blank lines, comments starting with ';', or
                # lines that begin with a digit (format artefacts).
                if (
                    not token
                    or token.startswith(";")
                    or token[0].isdigit()
                ):
                    continue
                if token not in self._stop_words:
                    words.add(token)

        logger.info(
            "SentimentDictLoader: %d %s-words loaded (after stop-word filter).",
            len(words),
            label,
        )
        return frozenset(words)

    def _load(self) -> Tuple[FrozenSet[str], FrozenSet[str]]:
        """
        Load both positive and negative word files.

        Returns
        -------
        Tuple[FrozenSet[str], FrozenSet[str]]
            ``(positive_words, negative_words)``
        """
        pos = self._load_word_file(
            self._config.master_dict_dir / "positive-words.txt",
            label="positive",
        )
        neg = self._load_word_file(
            self._config.master_dict_dir / "negative-words.txt",
            label="negative",
        )
        return pos, neg


# ===========================================================================
# ④  TextTokenizer
#    ─────────────
#    Responsible for converting a raw article string into clean sentence
#    and word token lists ready for downstream analysers.
# ===========================================================================
class TextTokenizer:
    """
    Two-level text tokenizer: sentence-level and word-level.

    Responsibilities
    ----------------
    1. Sentence tokenisation using NLTK's Punkt sentence boundary detector.
    2. Word tokenisation using NLTK's word tokenizer, retaining only
       alphabetic–alphanumeric tokens and discarding punctuation artefacts.
    3. Stop-word filtering to produce a "cleaned" word list for metrics that
       require it (Word Count, Subjectivity Score).

    Parameters
    ----------
    stop_words : FrozenSet[str]
        Combined stop-word set from :class:`StopWordLoader`.

    Notes
    -----
    Two word lists are produced to support metrics that need different
    granularities:

    * ``raw_words``     — all alphanumeric tokens, no stop-word filter,
                          used for readability / syllable calculations.
    * ``cleaned_words`` — stop-word filtered, used for Word Count and
                          Subjectivity Score.
    """

    # Pre-compiled pattern: keep only tokens that consist purely of word
    # characters (letters, digits, underscore).  This strips punctuation
    # tokens such as ``','``, ``'.'``, ``'—'`` that NLTK emits.
    _WORD_PATTERN: re.Pattern = re.compile(r"^\w+$")

    def __init__(self, stop_words: FrozenSet[str]) -> None:
        self._stop_words: FrozenSet[str] = stop_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(
        self,
        text: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Tokenise *text* into sentences, raw words, and cleaned words.

        Parameters
        ----------
        text : str
            Full article content as a single string.

        Returns
        -------
        sentences : List[str]
            List of sentence strings produced by NLTK's Punkt tokeniser.
        raw_words : List[str]
            All alphanumeric word tokens; punctuation excluded.
        cleaned_words : List[str]
            Subset of *raw_words* with stop-words removed (lower-cased).
        """
        if not text or not text.strip():
            return [], [], []

        # ── Sentence tokens ─────────────────────────────────────────────
        sentences: List[str] = nltk.sent_tokenize(text)

        # ── Word tokens ──────────────────────────────────────────────────
        all_tokens: List[str] = nltk.word_tokenize(text)
        raw_words: List[str] = [
            tok for tok in all_tokens
            if self._WORD_PATTERN.match(tok)
        ]

        # ── Cleaned words (stop-word filter applied) ──────────────────
        cleaned_words: List[str] = [
            tok.lower() for tok in raw_words
            if tok.lower() not in self._stop_words
        ]

        return sentences, raw_words, cleaned_words


# ===========================================================================
# ⑤  SentimentAnalyzer
#    ──────────────────
#    Computes all four sentiment variables defined in Text Analysis.docx §3.
# ===========================================================================
class SentimentAnalyzer:
    """
    Compute sentiment scores for a tokenised article.

    Variables Produced
    ------------------
    POSITIVE SCORE
        Sum of +1 for each cleaned word found in the positive dictionary.
    NEGATIVE SCORE
        Sum of +1 for each cleaned word found in the negative dictionary
        (stored as a positive integer per the specification).
    POLARITY SCORE
        ``(Positive − Negative) / ((Positive + Negative) + ε)``
        Range: ``[−1, +1]``.
    SUBJECTIVITY SCORE
        ``(Positive + Negative) / (Total Cleaned Words + ε)``
        Range: ``[0, +1]``.

    The epsilon value ``ε = 0.000001`` prevents division-by-zero and is
    taken verbatim from the assignment specification.

    Parameters
    ----------
    positive_words : FrozenSet[str]
        Positive sentiment lexicon from :class:`SentimentDictLoader`.
    negative_words : FrozenSet[str]
        Negative sentiment lexicon from :class:`SentimentDictLoader`.
    """

    # Epsilon used in all division operations to avoid division-by-zero.
    _EPSILON: float = 0.000001

    def __init__(
        self,
        positive_words: FrozenSet[str],
        negative_words: FrozenSet[str],
    ) -> None:
        self._pos_dict: FrozenSet[str] = positive_words
        self._neg_dict: FrozenSet[str] = negative_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        cleaned_words: List[str],
    ) -> Dict[str, float]:
        """
        Compute the four sentiment scores from a list of cleaned tokens.

        Parameters
        ----------
        cleaned_words : List[str]
            Lower-cased, stop-word-filtered word tokens from
            :class:`TextTokenizer`.

        Returns
        -------
        Dict[str, float]
            Keys: ``POSITIVE SCORE``, ``NEGATIVE SCORE``,
                  ``POLARITY SCORE``, ``SUBJECTIVITY SCORE``.
        """
        # ── Raw counts ───────────────────────────────────────────────────
        positive_score: int = sum(
            1 for word in cleaned_words if word in self._pos_dict
        )
        negative_score: int = sum(
            1 for word in cleaned_words if word in self._neg_dict
        )
        total_cleaned: int = len(cleaned_words)

        # ── Derived scores (formulae from Text Analysis.docx) ─────────
        polarity_score: float = (
            (positive_score - negative_score)
            / ((positive_score + negative_score) + self._EPSILON)
        )
        subjectivity_score: float = (
            (positive_score + negative_score)
            / (total_cleaned + self._EPSILON)
        )

        return {
            "POSITIVE SCORE":     positive_score,
            "NEGATIVE SCORE":     negative_score,
            "POLARITY SCORE":     round(polarity_score, 6),
            "SUBJECTIVITY SCORE": round(subjectivity_score, 6),
        }


# ===========================================================================
# ⑥  ReadabilityAnalyzer
#    ────────────────────
#    Implements the Gunning Fog Index and its required intermediate metrics.
# ===========================================================================
class ReadabilityAnalyzer:
    """
    Compute readability metrics using the Gunning Fog formula.

    Variables Produced
    ------------------
    AVG SENTENCE LENGTH
        ``Total word tokens / Total sentences``
    PERCENTAGE OF COMPLEX WORDS
        ``Complex word count / Total word tokens``
    FOG INDEX
        ``0.4 × (Avg Sentence Length + Percentage of Complex Words)``
    AVG NUMBER OF WORDS PER SENTENCE
        Identical to AVG SENTENCE LENGTH per the specification.

    Complex words are defined as words that contain **more than two
    syllables** (strictly ``> 2``).  Syllable counting is delegated to
    :class:`WordMetricsAnalyzer` to avoid duplicating logic.

    Parameters
    ----------
    syllable_counter : callable
        A callable that accepts a single word string and returns its
        syllable count as an integer.
    """

    # Gunning Fog constant (verbatim from specification)
    _FOG_CONSTANT: float = 0.4

    def __init__(self, syllable_counter) -> None:
        self._count_syllables = syllable_counter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        raw_words: List[str],
        sentences: List[str],
    ) -> Dict[str, float]:
        """
        Compute readability metrics.

        Parameters
        ----------
        raw_words : List[str]
            All word tokens (no stop-word filter, punctuation excluded).
        sentences : List[str]
            Sentence token list from :class:`TextTokenizer`.

        Returns
        -------
        Dict[str, float]
            Keys: ``AVG SENTENCE LENGTH``,
                  ``PERCENTAGE OF COMPLEX WORDS``,
                  ``FOG INDEX``,
                  ``AVG NUMBER OF WORDS PER SENTENCE``.
        """
        # Guard: empty text — return zeros to avoid ZeroDivisionError.
        if not raw_words or not sentences:
            return {
                "AVG SENTENCE LENGTH":         0.0,
                "PERCENTAGE OF COMPLEX WORDS": 0.0,
                "FOG INDEX":                   0.0,
                "AVG NUMBER OF WORDS PER SENTENCE": 0.0,
            }

        total_words: int     = len(raw_words)
        total_sentences: int = max(len(sentences), 1)

        # Count complex words (syllable count strictly greater than 2)
        complex_count: int = sum(
            1 for w in raw_words if self._count_syllables(w) > 2
        )

        avg_sentence_length: float = total_words / total_sentences
        pct_complex: float         = complex_count / total_words
        fog_index: float           = self._FOG_CONSTANT * (
            avg_sentence_length + pct_complex
        )

        return {
            "AVG SENTENCE LENGTH":         round(avg_sentence_length, 6),
            "PERCENTAGE OF COMPLEX WORDS": round(pct_complex, 6),
            "FOG INDEX":                   round(fog_index, 6),
            "AVG NUMBER OF WORDS PER SENTENCE": round(avg_sentence_length, 6),
        }


# ===========================================================================
# ⑦  WordMetricsAnalyzer
#    ────────────────────
#    Computes word-level statistics: complex word count, cleaned word count,
#    syllable density, personal pronoun count, and average word length.
# ===========================================================================
class WordMetricsAnalyzer:
    """
    Word-level statistical analyser.

    Variables Produced
    ------------------
    COMPLEX WORD COUNT
        Number of words in *raw_words* with more than two syllables.
    WORD COUNT
        Number of words remaining after stop-word removal and
        punctuation stripping (i.e. ``len(cleaned_words)``).
    SYLLABLE PER WORD
        Mean syllable count across *raw_words*.
    PERSONAL PRONOUNS
        Regex-based count of ``I``, ``we``, ``my``, ``ours``, ``us``
        (case-sensitive for ``I``; case-insensitive for the rest,
        **except** the country abbreviation ``US`` is excluded).
    AVG WORD LENGTH
        ``Sum of character counts across raw_words / total raw_words``.

    Syllable Counting Rules (from Text Analysis.docx §7)
    -----
    1. Count the number of vowels (a, e, i, o, u) in the word.
    2. If the word ends in ``"es"`` or ``"ed"``, subtract one from the count
       (those endings typically do not form a separate syllable).
    3. Ensure a minimum syllable count of 1 for any non-empty word.

    Personal Pronoun Regex (from Text Analysis.docx §8)
    ----
    Pattern: ``\\b(I|we|my|ours|us)\\b``  (case-insensitive)
    Special case: the token ``"US"`` (all caps, referring to the country)
    is explicitly excluded from the count.
    """

    # ── Pronoun pattern ─────────────────────────────────────────────────
    # Word-boundary anchors ensure "I" is not matched inside "inside",
    # "us" is not matched inside "thus", etc.
    _PRONOUN_PATTERN: re.Pattern = re.compile(
        r"\b(I|we|my|ours|us)\b",
        flags=re.IGNORECASE,
    )
    _VOWELS: FrozenSet[str] = frozenset("aeiou")

    # ------------------------------------------------------------------
    # Public API — syllable counting
    # ------------------------------------------------------------------

    def count_syllables(self, word: str) -> int:
        """
        Count syllables in *word* using the vowel-counting heuristic.

        The algorithm (verbatim from Text Analysis.docx):
          1. Lowercase the word.
          2. If it ends in ``"es"`` or ``"ed"``, strip those two characters
             so the silent ending is not counted.
          3. Count vowel characters.
          4. Return at least 1 to avoid assigning 0 syllables to any word.

        Parameters
        ----------
        word : str
            A single word token (no spaces, no punctuation).

        Returns
        -------
        int
            Estimated syllable count (≥ 1 for non-empty input).

        Examples
        --------
        >>> analyzer = WordMetricsAnalyzer()
        >>> analyzer.count_syllables("complicated")
        4
        >>> analyzer.count_syllables("foxes")
        1          # "foxes" → strip "es" → "fox" → 1 vowel
        """
        if not word:
            return 0

        lowered: str = word.lower()

        # Strip silent endings before counting
        if lowered.endswith("es") and len(lowered) > 3:
            lowered = lowered[:-2]
        elif lowered.endswith("ed") and len(lowered) > 3:
            lowered = lowered[:-2]

        # Count vowels
        syllable_count: int = sum(1 for ch in lowered if ch in self._VOWELS)

        # Guarantee minimum of 1 syllable for any real word
        return max(syllable_count, 1)

    # ------------------------------------------------------------------
    # Public API — main analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        raw_words: List[str],
        cleaned_words: List[str],
        original_text: str,
    ) -> Dict[str, float]:
        """
        Compute the five word-level metrics.

        Parameters
        ----------
        raw_words : List[str]
            All alphanumeric tokens (punctuation stripped, no stop-word filter).
        cleaned_words : List[str]
            Stop-word-filtered, lower-cased tokens.
        original_text : str
            The raw article text (needed for accurate pronoun regex search).

        Returns
        -------
        Dict[str, float]
            Keys: ``COMPLEX WORD COUNT``, ``WORD COUNT``,
                  ``SYLLABLE PER WORD``, ``PERSONAL PRONOUNS``,
                  ``AVG WORD LENGTH``.
        """
        if not raw_words:
            return {
                "COMPLEX WORD COUNT": 0,
                "WORD COUNT":         0,
                "SYLLABLE PER WORD":  0.0,
                "PERSONAL PRONOUNS":  0,
                "AVG WORD LENGTH":    0.0,
            }

        total_raw: int = len(raw_words)

        # ── Syllable metrics ─────────────────────────────────────────────
        syllable_counts: List[int] = [
            self.count_syllables(w) for w in raw_words
        ]
        complex_word_count: int = sum(
            1 for sc in syllable_counts if sc > 2
        )
        syllable_per_word: float = sum(syllable_counts) / total_raw

        # ── Word count (cleaned) ─────────────────────────────────────────
        word_count: int = len(cleaned_words)

        # ── Personal pronouns ────────────────────────────────────────────
        # We search the original text so we capture multi-sentence context
        # and naturally handle case ("I" vs. sentence-initial capitalisation).
        # The country abbreviation "US" (all-caps) is explicitly excluded.
        pronoun_count: int = self._count_personal_pronouns(original_text)

        # ── Average word length ──────────────────────────────────────────
        avg_word_length: float = (
            sum(len(w) for w in raw_words) / total_raw
        )

        return {
            "COMPLEX WORD COUNT": complex_word_count,
            "WORD COUNT":         word_count,
            "SYLLABLE PER WORD":  round(syllable_per_word, 6),
            "PERSONAL PRONOUNS":  pronoun_count,
            "AVG WORD LENGTH":    round(avg_word_length, 6),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_personal_pronouns(self, text: str) -> int:
        """
        Count personal pronouns in *text* using the specification regex.

        Matches the tokens ``I``, ``we``, ``my``, ``ours``, ``us`` (any
        case) at word boundaries.  The token ``"US"`` (exactly, meaning
        the country abbreviation in all caps) is deliberately excluded.

        Parameters
        ----------
        text : str
            Raw article text.

        Returns
        -------
        int
            Total count of personal pronoun occurrences.
        """
        matches: List[str] = self._PRONOUN_PATTERN.findall(text)
        # Remove the all-caps "US" country abbreviation
        filtered: List[str] = [m for m in matches if m != "US"]
        return len(filtered)


# ===========================================================================
# ⑧  ArticleAnalyzer
#    ────────────────
#    Orchestrator for a single article: reads the text file, runs all five
#    analysis stages, and returns a flat metrics dictionary.
# ===========================================================================
class ArticleAnalyzer:
    """
    Orchestrates the complete five-stage NLP analysis for a single article.

    This class is the glue that connects :class:`TextTokenizer`,
    :class:`SentimentAnalyzer`, :class:`ReadabilityAnalyzer`, and
    :class:`WordMetricsAnalyzer` into one coherent per-article workflow.

    Parameters
    ----------
    config : PipelineConfig
        Configuration object (used to build the article file path).
    tokenizer : TextTokenizer
        Pre-initialised tokenizer instance.
    sentiment_analyzer : SentimentAnalyzer
        Pre-initialised sentiment analyser instance.
    readability_analyzer : ReadabilityAnalyzer
        Pre-initialised readability analyser instance.
    word_metrics_analyzer : WordMetricsAnalyzer
        Pre-initialised word-metrics analyser instance.
    """

    # Ordered list of metric keys — guarantees consistent column ordering
    # in the output DataFrame regardless of dict insertion order.
    OUTPUT_COLUMNS: Tuple[str, ...] = (
        "POSITIVE SCORE",
        "NEGATIVE SCORE",
        "POLARITY SCORE",
        "SUBJECTIVITY SCORE",
        "AVG SENTENCE LENGTH",
        "PERCENTAGE OF COMPLEX WORDS",
        "FOG INDEX",
        "AVG NUMBER OF WORDS PER SENTENCE",
        "COMPLEX WORD COUNT",
        "WORD COUNT",
        "SYLLABLE PER WORD",
        "PERSONAL PRONOUNS",
        "AVG WORD LENGTH",
    )

    def __init__(
        self,
        config: PipelineConfig,
        tokenizer: TextTokenizer,
        sentiment_analyzer: SentimentAnalyzer,
        readability_analyzer: ReadabilityAnalyzer,
        word_metrics_analyzer: WordMetricsAnalyzer,
    ) -> None:
        self._config               = config
        self._tokenizer            = tokenizer
        self._sentiment_analyzer   = sentiment_analyzer
        self._readability_analyzer = readability_analyzer
        self._word_metrics_analyzer = word_metrics_analyzer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, url_id: str) -> Dict[str, object]:
        """
        Run the full NLP pipeline on the article identified by *url_id*.

        Execution Stages
        ----------------
        1. **Load** the ``.txt`` file from the scraped articles directory.
        2. **Tokenise** text into sentences, raw words, and cleaned words.
        3. **Sentiment** — compute four sentiment scores.
        4. **Readability** — compute Fog Index and derived metrics.
        5. **Word Metrics** — compute syllable, pronoun, and length stats.
        6. **Merge** all metric dictionaries into one ordered result dict.

        Parameters
        ----------
        url_id : str
            Unique identifier from ``Input.xlsx`` used as the filename stem
            for the corresponding scraped article (e.g., ``"TrackerOPS29012026"``).

        Returns
        -------
        Dict[str, object]
            A flat dictionary whose keys exactly match ``OUTPUT_COLUMNS``.
            If the article file is missing or unreadable, all values are 0.
        """
        text: str = self._load_text(url_id)

        if not text.strip():
            logger.warning(
                "ArticleAnalyzer: empty or missing text for URL_ID='%s'. "
                "All metrics will be zero.",
                url_id,
            )
            return self._zero_metrics()

        # Stage A – Tokenisation
        sentences, raw_words, cleaned_words = self._tokenizer.tokenize(text)

        # Stage B – Sentiment Analysis
        sentiment_metrics: Dict = self._sentiment_analyzer.analyze(cleaned_words)

        # Stage C – Readability
        readability_metrics: Dict = self._readability_analyzer.analyze(
            raw_words, sentences
        )

        # Stage D & E – Word Metrics (includes pronouns)
        word_metrics: Dict = self._word_metrics_analyzer.analyze(
            raw_words, cleaned_words, text
        )

        # Merge all metric dictionaries
        merged: Dict[str, object] = {
            **sentiment_metrics,
            **readability_metrics,
            **word_metrics,
        }

        # Return in canonical column order
        return {col: merged.get(col, 0) for col in self.OUTPUT_COLUMNS}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_text(self, url_id: str) -> str:
        """
        Read and return the article text for *url_id*.

        Parameters
        ----------
        url_id : str
            Article identifier (matches the stem of the ``.txt`` file).

        Returns
        -------
        str
            Full article content, or an empty string if the file is not found.
        """
        filepath: Path = self._config.scraped_dir / f"{url_id}.txt"
        if not filepath.exists():
            logger.warning(
                "ArticleAnalyzer: article file not found — '%s'", filepath
            )
            return ""
        try:
            return filepath.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logger.error(
                "ArticleAnalyzer: could not read '%s' — %s", filepath, exc
            )
            return ""

    def _zero_metrics(self) -> Dict[str, object]:
        """
        Return a zero-valued metrics dictionary for missing / empty articles.

        Returns
        -------
        Dict[str, object]
            All ``OUTPUT_COLUMNS`` mapped to ``0``.
        """
        return {col: 0 for col in self.OUTPUT_COLUMNS}


# ===========================================================================
# ⑨  NLPPipeline
#    ─────────────
#    Top-level driver.  Instantiates all components, reads the input
#    spreadsheet, iterates over every article with a tqdm progress bar,
#    and writes the final output to Excel.
# ===========================================================================
class NLPPipeline:
    """
    End-to-end NLP pipeline orchestrator.

    This class wires together all components of the analysis system,
    reads the ``Input.xlsx`` spreadsheet, processes every article in
    sequence with real-time tqdm progress reporting, and persists the
    structured results to ``Final_NLP_Output.xlsx``.

    Parameters
    ----------
    config : PipelineConfig, optional
        Custom configuration.  If not provided, the default
        ``PipelineConfig()`` is used (pointing to the standard
        parent directory workspace).

    Typical Usage
    -------------
    >>> pipeline = NLPPipeline()
    >>> pipeline.run()

    Architecture
    ------------
    NLPPipeline
    ├── PipelineConfig          ← filesystem paths
    ├── StopWordLoader          ← merged stop-word set
    ├── SentimentDictLoader     ← positive / negative lexicons
    ├── TextTokenizer           ← sentence & word tokenisation
    ├── WordMetricsAnalyzer     ← syllable counter (shared)
    ├── SentimentAnalyzer       ← four sentiment scores
    ├── ReadabilityAnalyzer     ← fog index family
    └── ArticleAnalyzer         ← per-article orchestrator
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config: PipelineConfig = config or PipelineConfig()
        self._article_analyzer: Optional[ArticleAnalyzer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the complete NLP pipeline end-to-end.

        Execution Steps
        ---------------
        1. **Bootstrap** — instantiate all components (logged).
        2. **Load Input** — read ``Input.xlsx`` to get URL_ID → URL pairs.
        3. **Analyse Articles** — iterate with tqdm progress bar; call
           :meth:`ArticleAnalyzer.analyze` for each URL_ID.
        4. **Assemble DataFrame** — merge input columns with metrics columns.
        5. **Export** — write to ``Final_NLP_Output.xlsx``.

        Returns
        -------
        pd.DataFrame
            The complete output DataFrame (also written to disk).

        Raises
        ------
        FileNotFoundError
            If ``Input.xlsx`` cannot be found (caught from ``PipelineConfig``).
        RuntimeError
            If the output file cannot be written to disk (e.g., open in Excel).
        """
        logger.info("=" * 68)
        logger.info("  NLP Text Analysis Pipeline — starting")
        logger.info("=" * 68)

        # Step 1: Bootstrap all components
        self._bootstrap()

        # Step 2: Load input spreadsheet
        input_df: pd.DataFrame = self._load_input()

        # Step 3: Analyse each article
        results: List[Dict] = self._analyse_all(input_df)

        # Step 4: Build output DataFrame
        output_df: pd.DataFrame = self._build_output(input_df, results)

        # Step 5: Persist to Excel
        self._export(output_df)

        logger.info("Pipeline complete.  Processed %d articles.", len(output_df))
        return output_df

    # ------------------------------------------------------------------
    # Private — pipeline stages
    # ------------------------------------------------------------------

    def _bootstrap(self) -> None:
        """
        Instantiate and wire all pipeline components.

        The construction order matters:
        StopWordLoader → SentimentDictLoader → TextTokenizer →
        WordMetricsAnalyzer → SentimentAnalyzer → ReadabilityAnalyzer →
        ArticleAnalyzer
        """
        logger.info("[1/5] Bootstrapping pipeline components …")

        # Load stop-words (merged from all 7 domain-specific files)
        with tqdm(total=1, desc="  Loading stop-words", unit="corpus") as pbar:
            stop_word_loader = StopWordLoader(self.config)
            stop_words: FrozenSet[str] = stop_word_loader.stop_words
            pbar.update(1)

        # Load positive / negative lexicons (stop-word filtered)
        with tqdm(total=1, desc="  Loading sentiment dict", unit="dict") as pbar:
            sent_loader = SentimentDictLoader(self.config, stop_words)
            pbar.update(1)

        # Instantiate analysers
        tokenizer            = TextTokenizer(stop_words)
        word_metrics_analyzer = WordMetricsAnalyzer()
        sentiment_analyzer   = SentimentAnalyzer(
            sent_loader.positive_words,
            sent_loader.negative_words,
        )
        readability_analyzer = ReadabilityAnalyzer(
            syllable_counter=word_metrics_analyzer.count_syllables
        )

        # Assemble the per-article orchestrator
        self._article_analyzer = ArticleAnalyzer(
            config=self.config,
            tokenizer=tokenizer,
            sentiment_analyzer=sentiment_analyzer,
            readability_analyzer=readability_analyzer,
            word_metrics_analyzer=word_metrics_analyzer,
        )
        logger.info("[1/5] Bootstrap complete.")

    def _load_input(self) -> pd.DataFrame:
        """
        Read ``Input.xlsx`` and return a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``URL_ID``, ``URL`` (and any extra columns present).

        Raises
        ------
        SystemExit
            If the file cannot be parsed, the pipeline logs the error and
            exits with code 1 to signal failure to the calling shell.
        """
        logger.info("[2/5] Loading input spreadsheet from '%s' …", self.config.input_file)
        try:
            df: pd.DataFrame = pd.read_excel(
                self.config.input_file,
                dtype={"URL_ID": str},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read Input.xlsx: %s", exc)
            sys.exit(1)

        if "URL_ID" not in df.columns:
            logger.error(
                "Input.xlsx is missing the 'URL_ID' column.  "
                "Found columns: %s",
                df.columns.tolist(),
            )
            sys.exit(1)

        logger.info("[2/5] Loaded %d rows from Input.xlsx.", len(df))
        return df

    def _analyse_all(self, input_df: pd.DataFrame) -> List[Dict]:
        """
        Iterate over every row in *input_df* and call the article analyser.

        A tqdm progress bar is displayed with per-article ETA and the
        current ``URL_ID`` in the bar description.

        Parameters
        ----------
        input_df : pd.DataFrame
            The input spreadsheet DataFrame.

        Returns
        -------
        List[Dict]
            One metrics dictionary per row, in the same order as *input_df*.
        """
        logger.info("[3/5] Analysing %d articles …", len(input_df))
        results: List[Dict] = []

        with tqdm(
            total=len(input_df),
            desc="  Analysing articles",
            unit="article",
            ncols=90,
            colour="cyan",
        ) as pbar:
            for _, row in input_df.iterrows():
                url_id: str = str(row["URL_ID"]).strip()
                pbar.set_postfix_str(url_id, refresh=True)

                metrics: Dict = self._article_analyzer.analyze(url_id)
                results.append(metrics)
                pbar.update(1)

        logger.info("[3/5] Article analysis complete.")
        return results

    def _build_output(
        self,
        input_df: pd.DataFrame,
        results: List[Dict],
    ) -> pd.DataFrame:
        """
        Merge the input columns with the computed metrics into one DataFrame.

        The output column order follows the specification exactly:
        all input columns first, then the 13 metric columns in the order
        defined by :attr:`ArticleAnalyzer.OUTPUT_COLUMNS`.

        Parameters
        ----------
        input_df : pd.DataFrame
            Original input spreadsheet DataFrame.
        results : List[Dict]
            One metrics dict per article in the same row order.

        Returns
        -------
        pd.DataFrame
            Combined output DataFrame ready for export.
        """
        logger.info("[4/5] Assembling output DataFrame …")
        metrics_df: pd.DataFrame = pd.DataFrame(results)

        # Concatenate input columns and metrics columns side-by-side.
        # reset_index() ensures aligned integer indices before concat.
        output_df: pd.DataFrame = pd.concat(
            [input_df.reset_index(drop=True), metrics_df.reset_index(drop=True)],
            axis=1,
        )

        # ── Data-type enforcement ─────────────────────────────────────
        # Integer columns should not be stored as floats in Excel.
        int_cols = [
            "POSITIVE SCORE", "NEGATIVE SCORE",
            "COMPLEX WORD COUNT", "WORD COUNT", "PERSONAL PRONOUNS",
        ]
        for col in int_cols:
            if col in output_df.columns:
                output_df[col] = output_df[col].astype(int)

        logger.info(
            "[4/5] Output DataFrame built — shape %s.", output_df.shape
        )
        return output_df

    def _export(self, output_df: pd.DataFrame) -> None:
        """
        Write *output_df* to an Excel workbook at ``config.output_file``.

        The workbook is written with ``openpyxl`` (the default engine for
        ``.xlsx``) and basic auto-column-width formatting is applied to
        improve readability without external dependencies.

        Parameters
        ----------
        output_df : pd.DataFrame
            The fully assembled output DataFrame.

        Raises
        ------
        RuntimeError
            If the file cannot be written (e.g., it is open in Excel).
        """
        logger.info(
            "[5/5] Exporting results to '%s' …", self.config.output_file
        )
        try:
            with pd.ExcelWriter(
                self.config.output_file,
                engine="openpyxl",
            ) as writer:
                output_df.to_excel(writer, index=False, sheet_name="NLP Analysis")

                # ── Auto-width columns ────────────────────────────────
                ws = writer.sheets["NLP Analysis"]
                for col_cells in ws.columns:
                    max_len = max(
                        (len(str(cell.value)) for cell in col_cells if cell.value),
                        default=10,
                    )
                    ws.column_dimensions[
                        col_cells[0].column_letter
                    ].width = min(max_len + 4, 50)

        except PermissionError:
            raise RuntimeError(
                f"Cannot write to '{self.config.output_file}'.  "
                "Please close the file if it is open in Excel and retry."
            )
        logger.info(
            "[5/5] Export complete — '%s'", self.config.output_file
        )


# ===========================================================================
# Script entry point
# ===========================================================================
if __name__ == "__main__":
    """
    Execute the NLP pipeline when the script is run directly.

    Steps
    -----
    1. Instantiate :class:`PipelineConfig` (uses default paths).
    2. Pass config to :class:`NLPPipeline`.
    3. Call :meth:`NLPPipeline.run` to execute the full pipeline.
    4. Print a final confirmation table (first 5 rows) and summary stats.
    """
    print("\n" + "=" * 68)
    print("   Blackcoffer NLP Text Analysis Pipeline")
    print("   Computes 13 linguistic metrics per article")
    print("=" * 68 + "\n")

    # ── Instantiate and run ──────────────────────────────────────────
    config   = PipelineConfig()
    pipeline = NLPPipeline(config=config)
    result_df: pd.DataFrame = pipeline.run()

    # ── Print preview ────────────────────────────────────────────────
    print("\n" + "-" * 68)
    print("  Preview -- first 5 rows of output:")
    print("-" * 68)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(result_df.head())

    print("\n" + "-" * 68)
    print("  Numeric summary statistics:")
    print("-" * 68)
    metric_cols = list(ArticleAnalyzer.OUTPUT_COLUMNS)
    print(result_df[metric_cols].describe().round(4).to_string())

    print(
        f"\n  [OK] Output written to:\n    {config.output_file}\n"
    )
