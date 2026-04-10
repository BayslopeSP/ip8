"""
searcher.py — IP8 Patent Infringement Detection System
Executes SerpAPI Google searches for company and product queries,
deduplicates and filters URLs against the blacklist, and returns a clean URL list.
"""

import logging
from urllib.parse import urlparse

from serpapi import GoogleSearch

from config import (
    SERP_API_KEY,
    SERPAPI_ENGINE,
    SERPAPI_COUNTRY,
    SERPAPI_LANGUAGE,
    SERPAPI_NUM_RESULTS,
    BLACKLISTED_DOMAINS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def clean_url(raw_url: str) -> str:
    """
    Strip trailing slashes and whitespace from a URL string.

    Args:
        raw_url: The URL string to normalize.

    Returns:
        Normalized URL without trailing whitespace or slashes.
    """
    return raw_url.strip().rstrip("/")


def extract_domain_from_url(url: str) -> str:
    """
    Parse and return the lowercase netloc (domain) component from a URL.

    Args:
        url: A valid URL string.

    Returns:
        Lowercase domain string (e.g. 'www.example.com'), or empty string on parse failure.
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def is_url_blacklisted(url: str) -> bool:
    """
    Check whether a URL belongs to any domain in the blacklist.

    Checks for substring matches, so 'blog.company.com' is caught by 'blog'
    and 'en.wikipedia.org' is caught by 'wikipedia.org'.

    Args:
        url: The URL string to evaluate.

    Returns:
        True if the URL should be excluded, False if it is clean.
    """
    domain = extract_domain_from_url(url)
    full_url_lower = url.lower()
    for blacklisted_pattern in BLACKLISTED_DOMAINS:
        if blacklisted_pattern in domain or blacklisted_pattern in full_url_lower:
            return True
    return False


def deduplicate_urls(urls: list[str]) -> list[str]:
    """
    Remove duplicate URLs from a list while preserving insertion order.

    Args:
        urls: List of URL strings, potentially containing duplicates.

    Returns:
        Deduplicated list of URL strings in their original order.
    """
    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in urls:
        cleaned = clean_url(url)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_urls.append(cleaned)
    return unique_urls


def filter_urls_against_blacklist(urls: list[str]) -> list[str]:
    """
    Remove all URLs that match any entry in the domain blacklist.

    Args:
        urls: List of URL strings to filter.

    Returns:
        Filtered list containing only non-blacklisted URLs.
    """
    filtered = [url for url in urls if not is_url_blacklisted(url)]
    removed_count = len(urls) - len(filtered)
    if removed_count > 0:
        logger.info("Blacklist filter removed %d URLs", removed_count)
    return filtered


# =============================================================================
# SERPAPI SEARCH
# =============================================================================


def execute_single_serpapi_query(query: str) -> tuple[list[str], str]:
    """
    Run a single Google search query via SerpAPI and return organic result URLs.

    Args:
        query: A search query string.

    Returns:
        Tuple of (list of URL strings, error message string or empty string).
    """
    search_params = {
        "engine": SERPAPI_ENGINE,
        "q": query,
        "gl": SERPAPI_COUNTRY,
        "hl": SERPAPI_LANGUAGE,
        "num": SERPAPI_NUM_RESULTS,
        "api_key": SERP_API_KEY,
    }
    try:
        search = GoogleSearch(search_params)
        results = search.get_dict()

        # SerpAPI returns an 'error' key when the key is invalid or quota exceeded.
        api_error = results.get("error", "")
        if api_error:
            logger.error("SerpAPI returned error for %r: %s",
                         query[:60], api_error)
            return [], str(api_error)

        organic_results = results.get("organic_results", [])
        urls = [item.get("link", "")
                for item in organic_results if item.get("link")]
        logger.info("SerpAPI query returned %d results: %r",
                    len(urls), query[:60])
        return urls, ""
    except Exception as exc:
        logger.error("SerpAPI query failed for %r: %s", query[:60], exc)
        return [], str(exc)


def execute_all_search_queries(
    company_queries: list[str],
    product_queries: list[str],
) -> dict[str, list[str]]:
    """
    Run all company and product SerpAPI queries and collect raw results.

    Executes each query sequentially, aggregating all returned URLs into
    separate lists for company and product results before deduplication.

    Args:
        company_queries: List of company discovery query strings.
        product_queries: List of product discovery query strings.

    Returns:
        Dict with keys 'company_urls' and 'product_urls', each a list of raw URLs.
    """
    company_urls: list[str] = []
    product_urls: list[str] = []
    first_error: str = ""

    logger.info("Executing %d company queries...", len(company_queries))
    for query in company_queries:
        urls, error = execute_single_serpapi_query(query)
        company_urls.extend(urls)
        if error and not first_error:
            first_error = error

    logger.info("Executing %d product queries...", len(product_queries))
    for query in product_queries:
        urls, error = execute_single_serpapi_query(query)
        product_urls.extend(urls)
        if error and not first_error:
            first_error = error

    return {
        "company_urls": company_urls,
        "product_urls": product_urls,
        "first_error": first_error,
    }


def search_and_collect_candidate_urls(
    company_queries: list[str],
    product_queries: list[str],
) -> list[str]:
    """
    Execute all search queries, deduplicate results, and apply blacklist filtering.

    This is the primary public API for the searcher module. It orchestrates all
    SerpAPI calls, combines company and product URLs, deduplicates across both
    sets, removes blacklisted domains, and returns a unified clean URL list.

    Args:
        company_queries: List of EXACTLY 4 company discovery query strings.
        product_queries: List of EXACTLY 6 product discovery query strings.

    Returns:
        Deduplicated, blacklist-filtered list of candidate URLs for scraping.
    """
    raw_results = execute_all_search_queries(company_queries, product_queries)

    # Propagate any API-level error so callers can surface it to the user.
    api_error = raw_results.get("first_error", "")
    if api_error:
        raise RuntimeError(f"SerpAPI error: {api_error}")

    all_raw_urls = raw_results["company_urls"] + raw_results["product_urls"]
    logger.info("Total raw URLs collected: %d", len(all_raw_urls))

    deduplicated_urls = deduplicate_urls(all_raw_urls)
    logger.info("After deduplication: %d URLs", len(deduplicated_urls))

    clean_candidate_urls = filter_urls_against_blacklist(deduplicated_urls)
    logger.info("After blacklist filter: %d URLs ready for scraping",
                len(clean_candidate_urls))

    return clean_candidate_urls
