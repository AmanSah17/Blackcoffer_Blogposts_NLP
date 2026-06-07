"""
main.py
=======

Author: Aman Sah
Description: Centralized runner for the Blackcoffer Text Analysis Assignment.
             Executes the web scraping first, followed by the NLP pipeline.
"""
import logging
import sys

# Import the refactored modules
from Notebooks.scrape_blackcoffer import BlackcofferScraper, ScraperConfig
from Notebooks.nlp_pipeline import NLPPipeline, PipelineConfig

logger = logging.getLogger("main")

def main():
    logger.info("==================================================")
    logger.info("   Blackcoffer End-to-End Pipeline started")
    logger.info("==================================================")

    # 1. Run the Scraper
    logger.info("\n>>> PHASE 1: Web Scraping <<<")
    scraper_config = ScraperConfig()
    scraper = BlackcofferScraper(config=scraper_config)
    try:
        scraper.run()
    except Exception as e:
        logger.error(f"Scraper failed: {e}")
        sys.exit(1)

    # 2. Run the NLP Pipeline
    logger.info("\n>>> PHASE 2: NLP Analysis <<<")
    nlp_config = PipelineConfig()
    pipeline = NLPPipeline(config=nlp_config)
    try:
        result_df = pipeline.run()
        logger.info(f"Pipeline finished successfully. Processed {len(result_df)} records.")
    except Exception as e:
        logger.error(f"NLP Pipeline failed: {e}")
        sys.exit(1)

    logger.info("==================================================")
    logger.info("   All operations completed successfully!")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
