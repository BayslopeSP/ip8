"""
scraper.py — IP8 Patent Infringement Detection System
Headless Selenium scraper that loads each candidate URL, extracts body text,
and attempts to parse product launch/announcement dates from the page content.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from config import (
    SCRAPER_PAGE_LOAD_WAIT_SECONDS,
    SCRAPER_MAX_TEXT_CHARS,
    SCRAPER_IMPLICIT_WAIT_SECONDS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ScrapedPage:
    """Container for data extracted from a single scraped web page."""

    url: str
    page_text: str
    extracted_date: Optional[str]
    scrape_success: bool
    error_message: str = ""


# =============================================================================
# DATE EXTRACTION HELPERS
# =============================================================================

_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
)

_DATE_PATTERNS = [
    # ISO format: 2023-04-15
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    # Month YYYY: January 2023
    re.compile(rf"\b({_MONTH_NAMES})\s+(\d{{4}})\b", re.IGNORECASE),
    # Month DD, YYYY: April 15, 2023
    re.compile(
        rf"\b({_MONTH_NAMES})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.IGNORECASE),
    # DD Month YYYY: 15 April 2023
    re.compile(
        rf"\b(\d{{1,2}})\s+({_MONTH_NAMES})\s+(\d{{4}})\b", re.IGNORECASE),
    # MM/DD/YYYY or DD/MM/YYYY patterns including year
    re.compile(r"\b(\d{1,2}[/\-]\d{1,2}[/\-](20\d{2}|19\d{2}))\b"),
]

_LAUNCH_CONTEXT_PATTERNS = [
    re.compile(r"launched?\s+in\b", re.IGNORECASE),
    re.compile(r"\breleased?\b", re.IGNORECASE),
    re.compile(r"\bannounced?\b", re.IGNORECASE),
    re.compile(r"\bavailable\s+(?:from|since)\b", re.IGNORECASE),
    re.compile(r"\bintroduced?\b", re.IGNORECASE),
    re.compile(r"\bdebuted?\b", re.IGNORECASE),
    re.compile(r"\bshipped?\b", re.IGNORECASE),
]


def attempt_date_extraction_from_text(page_text: str) -> Optional[str]:
    """
    Scan page text for product launch or announcement date strings and return the first match.

    Prioritizes dates that appear within 200 characters of a launch-context keyword
    (e.g., 'launched in', 'released', 'announced'). Falls back to the first date pattern
    found anywhere in the text.

    Args:
        page_text: Raw text content of a scraped web page.

    Returns:
        A date string (e.g., '2021-03-15' or 'March 2021') if found, else None.
    """
    if not page_text:
        return None

    # Strategy 1: Look for dates near launch-context keywords
    for context_pattern in _LAUNCH_CONTEXT_PATTERNS:
        for context_match in context_pattern.finditer(page_text):
            start_pos = max(0, context_match.start() - 50)
            end_pos = min(len(page_text), context_match.end() + 200)
            surrounding_text = page_text[start_pos:end_pos]

            for date_pattern in _DATE_PATTERNS:
                date_match = date_pattern.search(surrounding_text)
                if date_match:
                    return date_match.group(0).strip()

    # Strategy 2: Return the first date-like pattern found in the whole text
    for date_pattern in _DATE_PATTERNS:
        date_match = date_pattern.search(page_text)
        if date_match:
            return date_match.group(0).strip()

    return None


def format_date(raw_date_string: Optional[str]) -> str:
    """
    Return a display-friendly date string, substituting 'Unknown' for None/empty values.

    Args:
        raw_date_string: A raw date string (possibly None or empty).

    Returns:
        The date string as-is if non-empty, otherwise 'Unknown'.
    """
    if not raw_date_string:
        return "Unknown"
    return raw_date_string.strip()


# =============================================================================
# SELENIUM DRIVER FACTORY
# =============================================================================


def create_headless_chrome_driver() -> webdriver.Chrome:
    """
    Initialize and return a headless Chrome WebDriver instance with safe options.

    Uses webdriver-manager to automatically download/cache the appropriate
    ChromeDriver binary for the installed Chrome version.

    Returns:
        A configured headless Selenium Chrome WebDriver.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option(
        "excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.implicitly_wait(SCRAPER_IMPLICIT_WAIT_SECONDS)
    return driver


# =============================================================================
# PAGE SCRAPER
# =============================================================================


def scrape_page_content(url: str, driver: webdriver.Chrome) -> ScrapedPage:
    """
    Load a single URL using Selenium, wait for page render, and extract body text.

    Waits SCRAPER_PAGE_LOAD_WAIT_SECONDS after page load to allow JavaScript
    content to render. Extracts up to SCRAPER_MAX_TEXT_CHARS characters from
    the visible body text. Attempts to extract a product launch date from the text.

    Args:
        url: The target URL to scrape.
        driver: An active Selenium Chrome WebDriver instance.

    Returns:
        A ScrapedPage dataclass containing url, page_text, extracted_date,
        scrape_success flag, and any error_message.
    """
    try:
        driver.get(url)
        time.sleep(SCRAPER_PAGE_LOAD_WAIT_SECONDS)

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except Exception:
            pass  # Proceed even if wait times out

        body_element = driver.find_element(By.TAG_NAME, "body")
        raw_body_text = body_element.text or ""

        truncated_text = raw_body_text[:SCRAPER_MAX_TEXT_CHARS]
        extracted_date = attempt_date_extraction_from_text(
            raw_body_text[:10000])

        logger.info(
            "Scraped %d chars from %r | date found: %s",
            len(truncated_text),
            url[:80],
            extracted_date or "None",
        )

        return ScrapedPage(
            url=url,
            page_text=truncated_text,
            extracted_date=extracted_date,
            scrape_success=True,
        )

    except Exception as exc:
        logger.warning("Failed to scrape %r: %s", url[:80], exc)
        return ScrapedPage(
            url=url,
            page_text="",
            extracted_date=None,
            scrape_success=False,
            error_message=str(exc),
        )


def scrape_all_candidate_urls(candidate_urls: list[str]) -> list[ScrapedPage]:
    """
    Scrape all candidate URLs sequentially using a single shared headless Chrome driver.

    Initializes one Chrome driver, iterates over all URLs calling scrape_page_content()
    for each, and ensures the driver is properly closed afterward even if errors occur.
    Skips URLs that fail to load and logs warnings.

    Args:
        candidate_urls: List of clean, blacklist-filtered URLs to scrape.

    Returns:
        List of ScrapedPage objects for all attempted URLs (successful and failed).
    """
    scraped_pages: list[ScrapedPage] = []

    if not candidate_urls:
        logger.warning("No candidate URLs to scrape.")
        return scraped_pages

    logger.info("Starting sequential scrape of %d URLs...",
                len(candidate_urls))
    driver = None
    try:
        driver = create_headless_chrome_driver()
        for idx, url in enumerate(candidate_urls, start=1):
            logger.info("Scraping URL %d/%d: %s", idx,
                        len(candidate_urls), url[:80])
            page_result = scrape_page_content(url, driver)
            scraped_pages.append(page_result)
    finally:
        if driver:
            try:
                driver.quit()
                logger.info("Chrome driver closed successfully.")
            except Exception as quit_exc:
                logger.warning("Error closing Chrome driver: %s", quit_exc)

    successful_count = sum(1 for p in scraped_pages if p.scrape_success)
    logger.info(
        "Scraping complete: %d/%d pages successfully scraped.",
        successful_count,
        len(scraped_pages),
    )
    return scraped_pages
