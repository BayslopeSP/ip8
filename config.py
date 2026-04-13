"""
config.py — IP8 Patent Infringement Detection System
Global configuration: API keys, constants, blacklist, and app-wide settings.
"""

import os

from dotenv import load_dotenv

LANGFUSE_SECRET_KEY = "sk-lf-fe9b90cc-0e09-48aa-a9d6-d624280b8224"
LANGFUSE_PUBLIC_KEY = "pk-lf-7da81a25-5188-4817-86d7-7bf6cb8fccbc"
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"

# Load variables from .env file in the same directory (or any parent directory).
# override=True ensures .env values always win over stale system environment variables.
load_dotenv(override=True)

# =============================================================================
# API KEYS  (set in .env — never hard-code or commit real keys)
# =============================================================================

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
SERP_API_KEY: str = os.environ.get("SERP_API_KEY", "")

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.0
LLM_SEED: int = 42
LLM_MAX_TOKENS_DEFAULT: int = 2048
LLM_MAX_TOKENS_ANALYSIS: int = 1024

# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

SERPAPI_ENGINE: str = "google"
SERPAPI_COUNTRY: str = "us"
SERPAPI_LANGUAGE: str = "en"
SERPAPI_NUM_RESULTS: int = 10

COMPANY_QUERY_COUNT: int = 4
PRODUCT_QUERY_COUNT: int = 6

# =============================================================================
# SCRAPING CONFIGURATION
# =============================================================================

SCRAPER_PAGE_LOAD_WAIT_SECONDS: int = 3
SCRAPER_MAX_TEXT_CHARS: int = 3000
SCRAPER_IMPLICIT_WAIT_SECONDS: int = 10
SCRAPER_REQUEST_TIMEOUT_SECONDS: int = 15

GOOGLE_PATENTS_BASE_URL: str = "https://patents.google.com/patent/{patent_number}/en"

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

CACHE_DIRECTORY: str = ".ip8_cache"
CACHE_FILE_EXTENSION: str = ".pkl"

# =============================================================================
# INFRINGEMENT SCORING THRESHOLDS
# =============================================================================

SCORE_HIGH_THRESHOLD: float = 0.75
SCORE_MEDIUM_THRESHOLD: float = 0.40
SCORE_LABEL_HIGH: str = "HIGH"
SCORE_LABEL_MEDIUM: str = "MEDIUM"
SCORE_LABEL_LOW: str = "LOW"

MINIMUM_MATCHED_ELEMENTS_FOR_TECHNOLOGY_MATCH: int = 3

# =============================================================================
# URL BLACKLIST
# =============================================================================

BLACKLISTED_DOMAINS: list[str] = [
    "facebook.com",
    "linkedin.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "youtube.com",
    "wikipedia.org",
    "reddit.com",
    "quora.com",
    "medium.com",
    "blog",
    "techradar.com",
    "patents.google.com",
    "google.com/patents",
]

# =============================================================================
# GEOGRAPHIC SCOPE
# =============================================================================

ANALYSIS_SCOPE: str = "United States"
ANALYSIS_SCOPE_SHORT: str = "US"

# =============================================================================
# UI CONFIGURATION
# =============================================================================

APP_TITLE: str = "IP8 — Patent Infringement Detection System"
APP_ICON: str = "🔍"
APP_LAYOUT: str = "wide"

BADGE_HIGH_COLOR: str = "#FF4B4B"
BADGE_MEDIUM_COLOR: str = "#FFA500"
BADGE_LOW_COLOR: str = "#00CC66"

# =============================================================================
# REQUEST HEADERS
# =============================================================================

DEFAULT_REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
