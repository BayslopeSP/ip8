"""
patent_fetcher.py — IP8 Patent Infringement Detection System
Fetches and parses patent details from Google Patents using requests + BeautifulSoup.
Extracts bibliographic data, claims, description, CPC codes, and inventors.
"""

import re
import logging
from typing import Optional
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

from config import (
    GOOGLE_PATENTS_BASE_URL,
    DEFAULT_REQUEST_HEADERS,
    SCRAPER_REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class PatentData:
    """Container for all bibliographic and textual data extracted from a patent page."""

    patent_number: str = ""
    title: str = ""
    abstract: str = ""
    priority_date: str = ""
    filing_date: str = ""
    publication_date: str = ""
    assignee: str = ""
    inventors: list[str] = field(default_factory=list)
    cpc_codes: list[dict[str, str]] = field(default_factory=list)
    full_claims_text: str = ""
    description_text: str = ""
    raw_html: str = ""


# =============================================================================
# HELPERS
# =============================================================================


def normalize_patent_number(raw_patent_number: str) -> str:
    """
    Normalize a patent number by stripping whitespace and converting to uppercase.

    Args:
        raw_patent_number: User-supplied patent number string.

    Returns:
        Cleaned, uppercase patent number.
    """
    return raw_patent_number.strip().upper()


def build_google_patents_url(patent_number: str) -> str:
    """
    Build the full Google Patents URL for a given patent number.

    Args:
        patent_number: Normalized patent number (e.g. 'US10696212B2').

    Returns:
        Complete Google Patents URL string.
    """
    return GOOGLE_PATENTS_BASE_URL.format(patent_number=patent_number)


def truncate_text(text: str, max_chars: int = 5000) -> str:
    """
    Truncate a text string to a maximum character length, appending an ellipsis if cut.

    Args:
        text: The input text to possibly truncate.
        max_chars: Maximum number of characters to retain.

    Returns:
        Truncated text, with '...' appended if truncation occurred.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def clean_whitespace(text: str) -> str:
    """
    Collapse multiple consecutive whitespace characters into a single space and strip.

    Args:
        text: Raw text that may contain newlines, tabs, or extra spaces.

    Returns:
        Cleaned, whitespace-normalized string.
    """
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_element(element) -> str:
    """
    Safely extract and clean all text content from a BeautifulSoup Tag or None.

    Args:
        element: A BeautifulSoup Tag object, or None if the element was not found.

    Returns:
        Cleaned text string, or an empty string if the element is None.
    """
    if element is None:
        return ""
    return clean_whitespace(element.get_text(separator=" "))


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================


def _parse_title(soup: BeautifulSoup) -> str:
    """
    Extract the patent title from the Google Patents page.

    Args:
        soup: Parsed BeautifulSoup document.

    Returns:
        Patent title string, or empty string if not found.
    """
    title_tag = soup.find("span", {"itemprop": "title"})
    if title_tag:
        return clean_whitespace(title_tag.get_text())
    h1_tag = soup.find("h1", {"id": "title"})
    if h1_tag:
        return clean_whitespace(h1_tag.get_text())
    return ""


def _parse_abstract(soup: BeautifulSoup) -> str:
    """
    Extract the patent abstract from the Google Patents page.

    Args:
        soup: Parsed BeautifulSoup document.

    Returns:
        Abstract text string, or empty string if not found.
    """
    abstract_section = soup.find("section", {"itemprop": "abstract"})
    if abstract_section:
        return extract_text_from_element(abstract_section)
    abstract_div = soup.find("div", {"class": "abstract"})
    if abstract_div:
        return extract_text_from_element(abstract_div)
    return ""


def _parse_priority_date(soup: BeautifulSoup) -> str:
    """
    Extract the priority date from the bibliographic section of the patent page.

    Args:
        soup: Parsed BeautifulSoup document.

    Returns:
        Priority date string (e.g. '2018-03-15'), or empty string if not found.
    """
    priority_tag = soup.find("time", {"itemprop": "priorityDate"})
    if priority_tag:
        return priority_tag.get("datetime", priority_tag.get_text()).strip()

    # Fallback: search for a table row labeled 'Priority date'
    for label_tag in soup.find_all(["td", "th", "span", "div"]):
        label_text = label_tag.get_text(strip=True).lower()
        if "priority date" in label_text:
            sibling = label_tag.find_next_sibling()
            if sibling:
                return clean_whitespace(sibling.get_text())
    return ""


def _parse_filing_date(soup: BeautifulSoup) -> str:
    """
    Extract the filing date from the patent page.

    Args:
        soup: Parsed BeautifulSoup document.

    Returns:
        Filing date string, or empty string if not found.
    """
    filing_tag = soup.find("time", {"itemprop": "filingDate"})
    if filing_tag:
        return filing_tag.get("datetime", filing_tag.get_text()).strip()
    return ""


def _parse_publication_date(soup: BeautifulSoup) -> str:
    """
    Extract the publication date from the patent page.

    Args:
        soup: Parsed BeautifulSoup document.

    Returns:
        Publication date string, or empty string if not found.
    """
    pub_tag = soup.find("time", {"itemprop": "publicationDate"})
    if pub_tag:
        return pub_tag.get("datetime", pub_tag.get_text()).strip()
    return ""


def _parse_assignee(soup: BeautifulSoup) -> str:
    """
    Extract the patent assignee (owner organization) from the page.

    Args:
        soup: Parsed BeautifulSoup document.

    Returns:
        Assignee name string, or empty string if not found.
    """
    assignee_tag = soup.find("dd", {"itemprop": "assigneeOriginal"})
    if assignee_tag:
        return clean_whitespace(assignee_tag.get_text())
    assignee_span = soup.find("span", {"itemprop": "assigneeOriginal"})
    if assignee_span:
        return clean_whitespace(assignee_span.get_text())
    return ""


def _parse_inventors(soup: BeautifulSoup) -> list[str]:
    """
    Extract all inventor names listed on the patent page.

    Args:
        soup: Parsed BeautifulSoup document.

    Returns:
        List of inventor name strings.
    """
    inventors: list[str] = []
    for inventor_tag in soup.find_all(["dd", "span"], {"itemprop": "inventor"}):
        name = clean_whitespace(inventor_tag.get_text())
        if name and name not in inventors:
            inventors.append(name)
    return inventors


def _parse_cpc_codes(soup: BeautifulSoup) -> list[dict[str, str]]:
    """
    Extract Cooperative Patent Classification (CPC) codes and their descriptions.

    Args:
        soup: Parsed BeautifulSoup document.

    Returns:
        List of dicts, each with 'code' and 'label' keys.
    """
    cpc_entries: list[dict[str, str]] = []
    seen_codes: set[str] = set()

    for cpc_tag in soup.find_all(["span", "li"], {"itemprop": "cpcs"}):
        code_tag = cpc_tag.find("span", {"itemprop": "Code"})
        label_tag = cpc_tag.find("span", {"itemprop": "Description"})

        code = clean_whitespace(code_tag.get_text()) if code_tag else ""
        label = clean_whitespace(label_tag.get_text()) if label_tag else ""

        if code and code not in seen_codes:
            seen_codes.add(code)
            cpc_entries.append({"code": code, "label": label})

    # Broader fallback using classification links
    if not cpc_entries:
        for link in soup.find_all("a", {"data-cpc-code": True}):
            code = link.get("data-cpc-code", "").strip()
            label = clean_whitespace(link.get_text())
            if code and code not in seen_codes:
                seen_codes.add(code)
                cpc_entries.append({"code": code, "label": label})

    return cpc_entries


def _parse_claims_text(soup: BeautifulSoup) -> str:
    """
    Extract the full claims section text from the patent page.

    Args:
        soup: Parsed BeautifulSoup document.

    Returns:
        Full text of all patent claims, or empty string if not found.
    """
    claims_section = soup.find("section", {"itemprop": "claims"})
    if claims_section:
        return extract_text_from_element(claims_section)

    claims_div = soup.find("div", {"class": "claims"})
    if claims_div:
        return extract_text_from_element(claims_div)

    return ""


def _parse_description_text(soup: BeautifulSoup) -> str:
    """
    Extract the detailed description section text from the patent page.

    Args:
        soup: Parsed BeautifulSoup document.

    Returns:
        Description text (truncated to 8000 chars), or empty string if not found.
    """
    desc_section = soup.find("section", {"itemprop": "description"})
    if desc_section:
        raw_text = extract_text_from_element(desc_section)
        return truncate_text(raw_text, max_chars=8000)

    desc_div = soup.find("div", {"class": "description"})
    if desc_div:
        raw_text = extract_text_from_element(desc_div)
        return truncate_text(raw_text, max_chars=8000)

    return ""


# =============================================================================
# MAIN FETCH FUNCTION
# =============================================================================


def fetch_patent_details(patent_number: str) -> PatentData:
    """
    Fetch and parse all available data for a patent from Google Patents.

    Makes an HTTP GET request to the Google Patents page for the given patent
    number, then parses the HTML with BeautifulSoup to extract bibliographic
    metadata, claims, description, CPC codes, and inventor information.

    Args:
        patent_number: The patent identifier (e.g. 'US10696212B2').

    Returns:
        A PatentData dataclass populated with all extracted fields.

    Raises:
        requests.RequestException: If the HTTP request fails.
        ValueError: If patent_number is empty.
    """
    if not patent_number:
        raise ValueError("Patent number must not be empty.")

    normalized_number = normalize_patent_number(patent_number)
    url = build_google_patents_url(normalized_number)
    logger.info("Fetching patent data from: %s", url)

    response = requests.get(
        url,
        headers=DEFAULT_REQUEST_HEADERS,
        timeout=SCRAPER_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    patent_data = PatentData(
        patent_number=normalized_number,
        title=_parse_title(soup),
        abstract=_parse_abstract(soup),
        priority_date=_parse_priority_date(soup),
        filing_date=_parse_filing_date(soup),
        publication_date=_parse_publication_date(soup),
        assignee=_parse_assignee(soup),
        inventors=_parse_inventors(soup),
        cpc_codes=_parse_cpc_codes(soup),
        full_claims_text=_parse_claims_text(soup),
        description_text=_parse_description_text(soup),
        raw_html=response.text[:50000],  # Store first 50k chars for debug use
    )

    logger.info(
        "Patent fetch complete: title=%r, claims_len=%d, cpc_count=%d",
        patent_data.title,
        len(patent_data.full_claims_text),
        len(patent_data.cpc_codes),
    )

    return patent_data


def format_cpc_codes_for_display(cpc_codes: list[dict[str, str]]) -> str:
    """
    Format a list of CPC code dicts into a human-readable multi-line string.

    Args:
        cpc_codes: List of dicts with 'code' and 'label' keys.

    Returns:
        Formatted string like 'B60Q1/00 — Arrangement of optical signalling...'
    """
    if not cpc_codes:
        return "No CPC codes found"
    lines = []
    for entry in cpc_codes:
        code = entry.get("code", "")
        label = entry.get("label", "")
        if label:
            lines.append(f"{code} — {label}")
        else:
            lines.append(code)
    return "\n".join(lines)
