"""
scrape_blackcoffer.py
=====================

Author: Aman Sah
Description: An object-oriented scraper for Blackcoffer articles.
"""
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

logger = logging.getLogger(__name__)

@dataclass
class ScraperConfig:
    """Configuration for the Blackcoffer Scraper."""
    # Resolve the root automatically to be the parent of the Notebooks directory.
    # This ensures that when the reviewer downloads the folder, it works out of the box.
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    
    assignment_dir: Path = field(init=False)
    input_file: Path = field(init=False)
    output_dir: Path = field(init=False)
    
    def __post_init__(self) -> None:
        self.assignment_dir = self.root / "20211030 Test Assignment"
        self.input_file = self.assignment_dir / "Input.xlsx"
        self.output_dir = self.assignment_dir / "Scraped_Articles"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found at {self.input_file}")


class BlackcofferScraper:
    """
    Object-oriented article scraper using Trafilatura, BeautifulSoup4, and Selenium.
    """
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

    def __init__(self, config: ScraperConfig = None):
        self.config = config or ScraperConfig()

    def _clean_text(self, text: str) -> str:
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
            if self.NOISE_PATTERNS.search(line):
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def _extract_metadata_title(self, html: str, url: str) -> str:
        try:
            metadata = trafilatura.extract_metadata(html, default_url=url)
            if metadata and metadata.title:
                return self._clean_text(metadata.title)
        except Exception:
            pass
        soup = BeautifulSoup(html, "html.parser")
        for tag in ["h1", "title"]:
            node = soup.find(tag)
            if node and node.get_text(strip=True):
                return self._clean_text(node.get_text(" ", strip=True))
        return ""

    def _extract_with_trafilatura(self, html: str, url: str) -> str:
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

    def _extract_with_bs4(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for selector in self.NOISE_SELECTORS:
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
        return max(candidates, key=len) if candidates else soup.get_text("\n", strip=True)

    def _fetch_html_requests(self, url: str) -> str:
        response = requests.get(url, headers=self.HEADERS, timeout=30)
        response.raise_for_status()
        return response.text

    def _fetch_html_selenium(self, url: str) -> str:
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
        
        # Binary location is removed to improve cross-platform compatibility
        driver = webdriver.Chrome(options=options)
        try:
            driver.get(url)
            time.sleep(4)
            return driver.page_source
        finally:
            driver.quit()

    def _score_text(self, text: str) -> int:
        if not text:
            return 0
        words = re.findall(r"\w+", text)
        unique_words = len({word.lower() for word in words})
        return len(text) + unique_words * 2

    def extract_article(self, url: str) -> Tuple[str, str]:
        """Extracts the best possible title and text from a given URL."""
        html_sources = []
        
        try:
            html = trafilatura.fetch_url(url)
            if html:
                html_sources.append(("trafilatura_fetch", html))
        except Exception:
            pass

        if not html_sources:
            try:
                html_sources.append(("requests", self._fetch_html_requests(url)))
            except Exception:
                pass

        if not html_sources:
            try:
                html_sources.append(("selenium", self._fetch_html_selenium(url)))
            except Exception:
                pass

        best_title = ""
        best_text = ""
        best_score = -1

        for source_name, html in html_sources:
            title = self._extract_metadata_title(html, url)
            text = self._extract_with_trafilatura(html, url)
            if len(text) < 500:
                fallback_text = self._extract_with_bs4(html)
                if len(fallback_text) > len(text):
                    text = fallback_text
            text = self._clean_text(text)
            if title and text and not text.startswith(title):
                text = f"{title}\n\n{text}"
            score = self._score_text(text)
            if score > best_score:
                best_title = title
                best_text = text
                best_score = score

        if best_score < 500:
            try:
                html = self._fetch_html_selenium(url)
                title = self._extract_metadata_title(html, url)
                text = self._clean_text(self._extract_with_bs4(html))
                if title and text and not text.startswith(title):
                    text = f"{title}\n\n{text}"
                if self._score_text(text) > best_score:
                    best_title = title
                    best_text = text
            except Exception:
                pass

        return best_title, best_text

    def save_article(self, url_id: str, title: str, text: str) -> Path:
        """Saves the scraped article content to the disk."""
        output_path = self.config.output_dir / f"{url_id}.txt"
        content = text.strip()
        if title and content and not content.startswith(title):
            content = f"{title}\n\n{content}"
        output_path.write_text(content, encoding="utf-8")
        return output_path

    def run(self) -> pd.DataFrame:
        """Runs the complete scraping process based on the input excel file."""
        logger.info("Starting Blackcoffer Scraper...")
        input_df = pd.read_excel(self.config.input_file)
        results = []
        
        for row in tqdm(input_df.itertuples(index=False), total=len(input_df), desc="Scraping URLs"):
            url_id = str(row.URL_ID)
            url = str(row.URL)
            
            title, text = self.extract_article(url)
            saved_path = self.save_article(url_id, title, text)
            
            results.append({
                "URL_ID": url_id,
                "URL": url,
                "title": title,
                "chars": len(text),
                "output_file": str(saved_path),
            })

        results_df = pd.DataFrame(results)
        manifest_path = self.config.output_dir / "scrape_manifest.csv"
        results_df.to_csv(manifest_path, index=False)
        
        logger.info(f"Scraping complete. Saved {len(results_df)} articles to {self.config.output_dir}")
        return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = BlackcofferScraper()
    scraper.run()
