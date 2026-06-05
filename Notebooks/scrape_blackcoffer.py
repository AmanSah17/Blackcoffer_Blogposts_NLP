from pathlib import Path
import re
import time

import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

ROOT = Path(r"D:\gemma4\Test_assignment_20211030")
INPUT_FILE = ROOT / "20211030 Test Assignment" / "Input.xlsx"
OUTPUT_DIR = ROOT / "20211030 Test Assignment" / "Scraped_Articles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

NOISE_SELECTORS = [
    "script", "style", "noscript", "svg", "canvas", "iframe", "form",
    "nav", "footer", "header", "aside", "button", "input", "figure",
    "figcaption", "template", ".advertisement", ".ads", ".ad",
    ".adsbox", ".sponsor", ".sponsored",
]

NOISE_PATTERNS = re.compile(
    r"(cookie|subscribe|newsletter|advertis|sponsor|promoted|share this|"
    r"related posts|recommended|sign up|accept cookies|privacy policy|"
    r"terms of service|all rights reserved)",
    re.I,
)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []
    for line in lines:
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if NOISE_PATTERNS.search(line):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_metadata_title(html: str, url: str) -> str:
    try:
        metadata = trafilatura.extract_metadata(html, default_url=url)
        if metadata and metadata.title:
            return clean_text(metadata.title)
    except Exception:
        pass
    soup = BeautifulSoup(html, "html.parser")
    for tag in ["h1", "title"]:
        node = soup.find(tag)
        if node and node.get_text(strip=True):
            return clean_text(node.get_text(" ", strip=True))
    return ""


def extract_with_trafilatura(html: str, url: str) -> str:
    return trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_recall=True,
        deduplicate=True,
        with_metadata=False,
    ) or ""


def extract_with_bs4(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for selector in NOISE_SELECTORS:
        for node in soup.select(selector):
            node.decompose()

    noise_nodes = []
    for node in soup.find_all(True):
        attrs = " ".join(
            str(node.get(attr, "")) for attr in ("class", "id", "role", "aria-label")
        )
        if attrs and re.search(r"\b(ad|ads|advert|banner|promo|sponsor|cookie|popup|share|social|newsletter)\b", attrs, re.I):
            noise_nodes.append(node)
    for node in noise_nodes:
        node.decompose()

    candidates = []
    for selector in ["article", "main", "[role='main']", "#content", ".content", ".article", "body"]:
        node = soup.select_one(selector)
        if node:
            text = node.get_text("\n", strip=True)
            if text:
                candidates.append(text)
    text = max(candidates, key=len) if candidates else soup.get_text("\n", strip=True)
    return text


def fetch_html_requests(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def fetch_html_selenium(url: str) -> str:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=en-US")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        time.sleep(4)
        return driver.page_source
    finally:
        driver.quit()


def score_text(text: str) -> int:
    if not text:
        return 0
    words = re.findall(r"\w+", text)
    unique_words = len({word.lower() for word in words})
    return len(text) + unique_words * 2


def extract_article(url: str) -> tuple[str, str]:
    html_sources = []

    try:
        html = trafilatura.fetch_url(url)
        if html:
            html_sources.append(("trafilatura_fetch", html))
    except Exception:
        pass

    if not html_sources:
        try:
            html_sources.append(("requests", fetch_html_requests(url)))
        except Exception:
            pass

    if not html_sources:
        try:
            html_sources.append(("selenium", fetch_html_selenium(url)))
        except Exception:
            pass

    best_title = ""
    best_text = ""
    best_score = -1

    for source_name, html in html_sources:
        title = extract_metadata_title(html, url)
        text = extract_with_trafilatura(html, url)
        if len(text) < 500:
            fallback_text = extract_with_bs4(html)
            if len(fallback_text) > len(text):
                text = fallback_text
        text = clean_text(text)
        if title and text and not text.startswith(title):
            text = f"{title}\n\n{text}"
        score = score_text(text)
        if score > best_score:
            best_title = title
            best_text = text
            best_score = score

    if best_score < 500:
        try:
            html = fetch_html_selenium(url)
            title = extract_metadata_title(html, url)
            text = clean_text(extract_with_bs4(html))
            if title and text and not text.startswith(title):
                text = f"{title}\n\n{text}"
            if score_text(text) > best_score:
                best_title = title
                best_text = text
        except Exception:
            pass

    return best_title, best_text


def save_article(url_id: str, title: str, text: str) -> Path:
    output_path = OUTPUT_DIR / f"{url_id}.txt"
    content = text.strip()
    if title and content and not content.startswith(title):
        content = f"{title}\n\n{content}"
    output_path.write_text(content, encoding="utf-8")
    return output_path


input_df = pd.read_excel(INPUT_FILE)
results = []
for row in tqdm(input_df.itertuples(index=False), total=len(input_df), desc="Scraping URLs"):
    url_id = str(row.URL_ID)
    url = str(row.URL)
    title, text = extract_article(url)
    saved_path = save_article(url_id, title, text)
    results.append({
        "URL_ID": url_id,
        "URL": url,
        "title": title,
        "chars": len(text),
        "output_file": str(saved_path),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "scrape_manifest.csv", index=False)
print(results_df.head())
print(f"Saved {len(results_df)} articles to {OUTPUT_DIR}")
