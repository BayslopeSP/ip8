"""
analyzer.py — IP8 Patent Infringement Detection System
Analyzes each scraped product page against patent claim elements using LLM,
scores infringement likelihood, and classifies results into HIGH/MEDIUM/LOW buckets.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_SEED,
    LLM_MAX_TOKENS_ANALYSIS,
    SCORE_HIGH_THRESHOLD,
    SCORE_MEDIUM_THRESHOLD,
    SCORE_LABEL_HIGH,
    SCORE_LABEL_MEDIUM,
    SCORE_LABEL_LOW,
    MINIMUM_MATCHED_ELEMENTS_FOR_TECHNOLOGY_MATCH,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LLM CLIENT (singleton pattern)
# =============================================================================

_openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """
    Return a singleton OpenAI client, initializing it on first call.

    Returns:
        Configured OpenAI client instance.
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ProductAnalysisResult:
    """Structured output from a single LLM product-to-patent infringement analysis."""

    url: str
    company_name: str = ""
    product_name: str = ""
    technology_match: bool = False
    similarity_score: int = 0
    matched_elements: list[str] = field(default_factory=list)
    unmatched_elements: list[str] = field(default_factory=list)
    product_launch_date: str = ""
    launch_after_priority_date: Optional[bool] = None
    plain_english_analysis: str = ""
    is_product_page: bool = False
    infringement_score: float = 0.0
    infringement_label: str = SCORE_LABEL_LOW
    total_elements: int = 0
    matched_element_count: int = 0
    analysis_error: str = ""


# =============================================================================
# PROMPTS
# =============================================================================

_INFRINGEMENT_ANALYST_SYSTEM_PROMPT = """You are a senior patent infringement analyst and \
intellectual property attorney with expertise in claim construction and product-to-patent mapping. \
You systematically evaluate whether a real-world product implements the technology described in a \
patent claim. You are precise, evidence-based, and conservative — you only flag infringement when \
there is clear semantic alignment between the claim elements and the product features. You \
understand that infringement requires ALL elements of at least one independent claim to be present \
in the product. Always return valid JSON. Do not include markdown code fences or extra explanation."""

_PRODUCT_ANALYSIS_USER_PROMPT = """Evaluate whether the following product potentially infringes \
the given patent claim.

PATENT CONTEXT:
- Patent Title: {title}
- Priority Date: {priority_date}
- Novelty: {novelty_summary}
- Domain: {domain} | Sub-domain: {subdomain}
- Independent Claim Text: {selected_claim_text}
- Claim Elements to match:
{numbered_elements_list}

PRODUCT INFORMATION:
- Source URL: {url}
- Product launch/announcement date: {extracted_date}
- Page content: {page_text}

INSTRUCTIONS:
1. Evaluate if this page describes a real commercial product (not a blog/article)
2. For each claim element, determine if the product implements it semantically
3. A match requires functional equivalence, not identical wording
4. Only flag technology_match=true if at least {min_matched_elements} claim elements are present
5. Note: infringement only applies if product launch is AFTER {priority_date}

CRITICAL: In matched_elements and unmatched_elements, copy the EXACT text from the
numbered claim elements list above (e.g. "Sensor array — detects motion in 360° field").
Never write "element1", "element2", "element3" or any generic placeholder text.

Return ONLY valid JSON:
{{
  "company_name": "",
  "product_name": "",
  "technology_match": true,
  "similarity_score": 0,
  "matched_elements": ["<component — function, copied verbatim from the numbered list above>"],
  "unmatched_elements": ["<component — function, copied verbatim from the numbered list above>"],
  "product_launch_date": "",
  "launch_after_priority_date": null,
  "plain_english_analysis": "",
  "is_product_page": true
}}"""


# =============================================================================
# HELPERS
# =============================================================================


def _call_llm_for_json(system_prompt: str, user_prompt: str) -> dict:
    """
    Invoke LLM with given prompts and parse the text response as a JSON dict.

    Args:
        system_prompt: System-level instruction string.
        user_prompt: User-level context and query string.

    Returns:
        Parsed JSON dict from the LLM response.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
    """
    client = get_openai_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
        max_tokens=LLM_MAX_TOKENS_ANALYSIS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_text = response.choices[0].message.content.strip()
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)
    return json.loads(raw_text)


def build_numbered_elements_list(elements: list[dict[str, str]]) -> str:
    """
    Format claim elements as a numbered list string for prompt injection.

    Args:
        elements: List of dicts with 'component' and 'function' keys.

    Returns:
        Multi-line numbered string describing each element.
    """
    if not elements:
        return "  No elements specified."
    lines = []
    for idx, elem in enumerate(elements, start=1):
        component = elem.get("component", "")
        function = elem.get("function", "")
        lines.append(f"  {idx}. {component} — {function}")
    return "\n".join(lines)


def normalize_score(raw_score: float, total_elements: int) -> float:
    """
    Normalize an infringement score to a 0.0–1.0 float ratio.

    Clamps the result to [0.0, 1.0] to handle any edge cases in LLM output.

    Args:
        raw_score: Number of matched elements (numerator).
        total_elements: Total elements in the claim (denominator).

    Returns:
        Float in range [0.0, 1.0].
    """
    if total_elements <= 0:
        return 0.0
    return max(0.0, min(1.0, raw_score / total_elements))


def classify_infringement_level(infringement_score: float) -> str:
    """
    Map a 0.0–1.0 infringement score to a HIGH / MEDIUM / LOW label.

    Args:
        infringement_score: Normalized float ratio of matched to total elements.

    Returns:
        One of 'HIGH', 'MEDIUM', or 'LOW' as defined by the config thresholds.
    """
    if infringement_score >= SCORE_HIGH_THRESHOLD:
        return SCORE_LABEL_HIGH
    if infringement_score >= SCORE_MEDIUM_THRESHOLD:
        return SCORE_LABEL_MEDIUM
    return SCORE_LABEL_LOW


def truncate_page_text_for_prompt(page_text: str, max_chars: int = 2000) -> str:
    """
    Truncate page text to fit within the LLM prompt token budget.

    Args:
        page_text: Full scraped page text.
        max_chars: Maximum characters to include in the prompt.

    Returns:
        Truncated text with '...' suffix if cut short.
    """
    if len(page_text) <= max_chars:
        return page_text
    return page_text[:max_chars] + "..."


# =============================================================================
# INFRINGEMENT SCORING
# =============================================================================


def calculate_infringement_score(
    matched_elements: list[str],
    total_elements: int,
) -> tuple[float, str]:
    """
    Compute the final infringement score and label from matched vs. total elements.

    infringement_score = len(matched_elements) / total_elements
    Label is assigned per the config thresholds: HIGH ≥ 0.75, MEDIUM ≥ 0.40, else LOW.

    Args:
        matched_elements: List of claim element strings that were matched in the product.
        total_elements: Total number of claim elements being tested.

    Returns:
        Tuple of (infringement_score: float, label: str).
    """
    matched_count = len(matched_elements)
    score = normalize_score(matched_count, total_elements)
    label = classify_infringement_level(score)
    return score, label


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================


def analyze_product_for_infringement(
    scraped_page,
    patent_title: str,
    priority_date: str,
    novelty_summary: str,
    domain: str,
    subdomain: str,
    selected_claim_text: str,
    claim_elements: list[dict[str, str]],
) -> ProductAnalysisResult:
    """
    Use LLM to evaluate whether a scraped product page infringes the selected patent claim.

    Sends the patent context (title, priority date, novelty, domain, claim text, and
    decomposed elements) alongside the scraped page's URL, body text, and extracted date
    to GPT. Parses the structured JSON response and computes the final infringement score.

    Args:
        scraped_page: A ScrapedPage instance with url, page_text, and extracted_date.
        patent_title: The title of the patent being analyzed.
        priority_date: The patent priority date string for temporal infringement checking.
        novelty_summary: 2-3 sentence patent novelty description.
        domain: Technology domain classification string.
        subdomain: Technology subdomain classification string.
        selected_claim_text: Full text of the selected independent claim.
        claim_elements: List of decomposed claim element dicts.

    Returns:
        A fully populated ProductAnalysisResult dataclass.
    """
    result = ProductAnalysisResult(
        url=scraped_page.url,
        total_elements=len(claim_elements),
    )

    if not scraped_page.scrape_success or not scraped_page.page_text.strip():
        result.analysis_error = "Page scrape failed or empty content."
        return result

    numbered_elements = build_numbered_elements_list(claim_elements)
    extracted_date_str = scraped_page.extracted_date or "Unknown"
    truncated_page_text = truncate_page_text_for_prompt(
        scraped_page.page_text, max_chars=2000)

    user_prompt = _PRODUCT_ANALYSIS_USER_PROMPT.format(
        title=patent_title,
        priority_date=priority_date or "Unknown",
        novelty_summary=novelty_summary[:400],
        domain=domain,
        subdomain=subdomain,
        selected_claim_text=selected_claim_text[:1000],
        numbered_elements_list=numbered_elements,
        url=scraped_page.url,
        extracted_date=extracted_date_str,
        page_text=truncated_page_text,
        min_matched_elements=MINIMUM_MATCHED_ELEMENTS_FOR_TECHNOLOGY_MATCH,
    )

    try:
        llm_output = _call_llm_for_json(
            _INFRINGEMENT_ANALYST_SYSTEM_PROMPT, user_prompt)

        result.company_name = str(llm_output.get("company_name", ""))
        result.product_name = str(llm_output.get("product_name", ""))
        result.technology_match = bool(
            llm_output.get("technology_match", False))
        result.similarity_score = int(llm_output.get("similarity_score", 0))
        result.matched_elements = list(llm_output.get("matched_elements", []))
        result.unmatched_elements = list(
            llm_output.get("unmatched_elements", []))
        result.product_launch_date = str(
            llm_output.get("product_launch_date", ""))
        result.launch_after_priority_date = llm_output.get(
            "launch_after_priority_date")
        result.plain_english_analysis = str(
            llm_output.get("plain_english_analysis", ""))
        result.is_product_page = bool(llm_output.get("is_product_page", False))

        # Compute final infringement score from element match ratio
        result.matched_element_count = len(result.matched_elements)
        score, label = calculate_infringement_score(
            result.matched_elements,
            result.total_elements,
        )
        result.infringement_score = score
        result.infringement_label = label

        logger.info(
            "Analysis complete for %r — match=%s score=%.2f label=%s",
            scraped_page.url[:60],
            result.technology_match,
            result.infringement_score,
            result.infringement_label,
        )

    except Exception as exc:
        logger.error("LLM analysis failed for %r: %s",
                     scraped_page.url[:60], exc)
        result.analysis_error = str(exc)

    return result


def analyze_all_scraped_pages(
    scraped_pages: list,
    patent_title: str,
    priority_date: str,
    novelty_summary: str,
    domain: str,
    subdomain: str,
    selected_claim_text: str,
    claim_elements: list[dict[str, str]],
) -> list[ProductAnalysisResult]:
    """
    Run infringement analysis on all scraped pages and return filtered, sorted results.

    Processes each page sequentially, discards results where is_product_page=False
    or technology_match=False, and sorts the remaining results by infringement_score
    in descending order (highest infringement likelihood first).

    Args:
        scraped_pages: List of ScrapedPage objects from scraper.py.
        patent_title: Patent title string.
        priority_date: Patent priority date string.
        novelty_summary: Patent novelty description.
        domain: Technology domain.
        subdomain: Technology subdomain.
        selected_claim_text: Full text of the selected patent claim.
        claim_elements: Decomposed claim element list.

    Returns:
        Sorted list of ProductAnalysisResult objects (product pages with technology match only).
    """
    all_results: list[ProductAnalysisResult] = []

    logger.info("Analyzing %d scraped pages for infringement...",
                len(scraped_pages))
    for idx, page in enumerate(scraped_pages, start=1):
        logger.info("Analyzing page %d/%d: %s", idx,
                    len(scraped_pages), page.url[:60])
        analysis = analyze_product_for_infringement(
            scraped_page=page,
            patent_title=patent_title,
            priority_date=priority_date,
            novelty_summary=novelty_summary,
            domain=domain,
            subdomain=subdomain,
            selected_claim_text=selected_claim_text,
            claim_elements=claim_elements,
        )
        all_results.append(analysis)

    # Filter to only confirmed product pages with technology match
    qualifying_results = [
        r for r in all_results
        if r.is_product_page and r.technology_match and not r.analysis_error
    ]

    # Sort by infringement score descending
    qualifying_results.sort(key=lambda r: r.infringement_score, reverse=True)

    logger.info(
        "Analysis finished: %d qualifying results from %d total pages.",
        len(qualifying_results),
        len(all_results),
    )
    return qualifying_results
