import os
import re
import pandas as pd
from pathlib import Path
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

ROOT = Path(r"D:\gemma4\Test_assignment_20211030")
INPUT_FILE = ROOT / "20211030 Test Assignment" / "Input.xlsx"
OUTPUT_FILE = ROOT / "20211030 Test Assignment" / "Output Data Structure.xlsx"
SCRAPED_DIR = ROOT / "20211030 Test Assignment" / "Scraped_Articles"
STOPWORDS_DIR = ROOT / "20211030 Test Assignment" / "StopWords"
MASTER_DICT_DIR = ROOT / "20211030 Test Assignment" / "MasterDictionary"

def load_stopwords():
    stop_words = set()
    if not STOPWORDS_DIR.exists():
        return stop_words
    for filepath in STOPWORDS_DIR.glob("*.txt"):
        with open(filepath, "r", encoding="latin-1") as f:
            for line in f:
                word = line.split('|')[0].strip().lower()
                if word:
                    stop_words.add(word)
    return stop_words

def load_master_dict(stop_words):
    positive_words = set()
    negative_words = set()
    
    pos_file = MASTER_DICT_DIR / "positive-words.txt"
    if pos_file.exists():
        with open(pos_file, "r", encoding="latin-1") as f:
            for line in f:
                word = line.strip().lower()
                if word and word not in stop_words:
                    positive_words.add(word)
                    
    neg_file = MASTER_DICT_DIR / "negative-words.txt"
    if neg_file.exists():
        with open(neg_file, "r", encoding="latin-1") as f:
            for line in f:
                word = line.strip().lower()
                if word and word not in stop_words:
                    negative_words.add(word)
                    
    return positive_words, negative_words

def count_syllables(word):
    word = word.lower()
    if word.endswith("es"):
        word = word[:-2]
    elif word.endswith("ed"):
        word = word[:-2]
    
    vowels = "aeiou"
    count = 0
    for char in word:
        if char in vowels:
            count += 1
    # Ensure at least 1 syllable if the word has any characters
    if count == 0 and len(word) > 0:
        count = 1
    return count

def analyze_text(text, stop_words, positive_words, negative_words):
    # If text is empty, return default 0 values
    if not text or not text.strip():
        return {
            "POSITIVE SCORE": 0,
            "NEGATIVE SCORE": 0,
            "POLARITY SCORE": 0,
            "SUBJECTIVITY SCORE": 0,
            "AVG SENTENCE LENGTH": 0,
            "PERCENTAGE OF COMPLEX WORDS": 0,
            "FOG INDEX": 0,
            "AVG NUMBER OF WORDS PER SENTENCE": 0,
            "COMPLEX WORD COUNT": 0,
            "WORD COUNT": 0,
            "SYLLABLE PER WORD": 0,
            "PERSONAL PRONOUNS": 0,
            "AVG WORD LENGTH": 0
        }
        
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    
    # Remove punctuations and calculate total raw words
    raw_words = [w for w in words if re.match(r'^\w+$', w)]
    
    # Personal Pronouns (regex match before lowercasing)
    # Using word boundaries to capture "I", "we", "my", "ours", "us", excluding "US"
    pronoun_regex = re.compile(r'\b(I|we|my|ours|us)\b')
    personal_pronouns = 0
    for w in raw_words:
        if w == "US":
            continue
        if pronoun_regex.match(w):
            personal_pronouns += 1

    # Word count and cleaning using stopwords
    cleaned_words = [w.lower() for w in raw_words if w.lower() not in stop_words]
    word_count = len(cleaned_words)
    
    pos_score = 0
    neg_score = 0
    for w in cleaned_words:
        if w in positive_words:
            pos_score += 1
        if w in negative_words:
            neg_score += 1

    polarity_score = (pos_score - neg_score) / ((pos_score + neg_score) + 0.000001)
    subjectivity_score = (pos_score + neg_score) / (word_count + 0.000001)
    
    # Readability
    num_sentences = len(sentences) if len(sentences) > 0 else 1
    avg_sentence_length = len(raw_words) / num_sentences
    
    complex_word_count = 0
    syllable_count_total = 0
    word_chars_total = 0
    
    for w in raw_words:
        s_count = count_syllables(w)
        syllable_count_total += s_count
        if s_count > 2:
            complex_word_count += 1
        word_chars_total += len(w)
            
    percentage_complex_words = complex_word_count / len(raw_words) if len(raw_words) > 0 else 0
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    avg_words_per_sentence = len(raw_words) / num_sentences
    
    syllables_per_word = syllable_count_total / len(raw_words) if len(raw_words) > 0 else 0
    avg_word_length = word_chars_total / len(raw_words) if len(raw_words) > 0 else 0

    return {
        "POSITIVE SCORE": pos_score,
        "NEGATIVE SCORE": neg_score,
        "POLARITY SCORE": round(polarity_score, 4),
        "SUBJECTIVITY SCORE": round(subjectivity_score, 4),
        "AVG SENTENCE LENGTH": round(avg_sentence_length, 4),
        "PERCENTAGE OF COMPLEX WORDS": round(percentage_complex_words, 4),
        "FOG INDEX": round(fog_index, 4),
        "AVG NUMBER OF WORDS PER SENTENCE": round(avg_words_per_sentence, 4),
        "COMPLEX WORD COUNT": complex_word_count,
        "WORD COUNT": word_count,
        "SYLLABLE PER WORD": round(syllables_per_word, 4),
        "PERSONAL PRONOUNS": personal_pronouns,
        "AVG WORD LENGTH": round(avg_word_length, 4)
    }

def main():
    print("Loading stopwords...")
    stop_words = load_stopwords()
    print("Loading master dictionary...")
    positive_words, negative_words = load_master_dict(stop_words)
    
    print("Reading input file...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    results = []
    
    print("Processing articles...")
    for index, row in df.iterrows():
        url_id = str(row['URL_ID'])
        file_path = SCRAPED_DIR / f"{url_id}.txt"
        
        text = ""
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            print(f"Warning: Article file {url_id}.txt not found.")
            
        metrics = analyze_text(text, stop_words, positive_words, negative_words)
        
        # Combine input row with metrics
        row_dict = row.to_dict()
        row_dict.update(metrics)
        results.append(row_dict)

    print("Generating output data structure...")
    out_df = pd.DataFrame(results)
    
    # Ensure correct column order if Output Data Structure.xlsx exists
    # We will just write the columns in order they were in row_dict
    
    try:
        out_df.to_excel(OUTPUT_FILE, index=False)
        print(f"Output saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    main()
