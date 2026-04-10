"""
query_builder.py — IP8 Patent Infringement Detection System
Generates two sets of SerpAPI search queries from patent claim elements using LLM:
  Set A — Company discovery queries (4 queries)
  Set B — Product discovery queries (6 queries)
"""

import json
import logging
import re
from typing import Optional

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_SEED,
    LLM_MAX_TOKENS_DEFAULT,
    COMPANY_QUERY_COUNT,
    PRODUCT_QUERY_COUNT,
    ANALYSIS_SCOPE,
)

logger = logging.getLogger(__name__)

# =============================================================================
# LLM CLIENT (reuse singleton pattern)
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
# PROMPTS
# =============================================================================

_QUERY_BUILDER_SYSTEM_PROMPT = """You are an expert patent intelligence analyst and search engine \
optimization specialist. You create precise, expert-level Google search queries to identify \
companies and products that may be using technology described in a patent claim. Your queries \
must be specific, contextual, and optimized for Google Search API. Avoid generic terms. \
Think like a patent litigator building a prior art and infringement case. \
Always return valid JSON. Do not include markdown code fences or extra explanation."""

_QUERY_BUILDER_USER_PROMPT = """Generate search queries for patent infringement detection.

Patent context:
- Industry: {industry}
- Domain: {domain}
- Sub-domain: {subdomain}
- Claim elements: {elements_list}
- Technology novelty: {novelty_summary}
- Scope: {scope} companies and products only

Generate EXACTLY {company_query_count} company discovery queries (Set A) and EXACTLY \
{product_query_count} product discovery queries (Set B).

Set A queries must:
- Target US companies working on {domain} technology
- Reference specific technical functionality from the claim elements
- Use Boolean operators (OR, AND) where relevant

Set B queries must:
- Target specific products implementing the claimed technology
- Include known company names in the domain where applicable
- Reference the specific technical workflow of the claim
- Use quoted phrases for precision

Return JSON:
{{
  "company_queries": ["query1", "query2", "query3", "query4"],
  "product_queries": ["query1", "query2", "query3", "query4", "query5", "query6"]
}}"""


# =============================================================================
# HELPERS
# =============================================================================


def _call_llm_for_json(system_prompt: str, user_prompt: str) -> dict:
    """
    Invoke LLM with given prompts and parse the text response as a JSON dict.

    Args:
        system_prompt: System-level instruction string for the LLM.
        user_prompt: User-level context and query string.

    Returns:
        Parsed JSON dict from the LLM response.

    Raises:
        json.JSONDecodeError: If the LLM response is not valid JSON.
    """
    client = get_openai_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED,
        max_tokens=LLM_MAX_TOKENS_DEFAULT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_text = response.choices[0].message.content.strip()
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)
    return json.loads(raw_text)


def format_claim_elements_for_prompt(elements: list[dict[str, str]]) -> str:
    """
    Convert a list of claim element dicts into a numbered text list for prompt injection.

    Args:
        elements: List of dicts with 'component' and 'function' keys.

    Returns:
        Numbered multi-line string like:
        '1. optical sensor — detects ambient light levels...'
    """
    if not elements:
        return "No elements available"
    lines = []
    for idx, elem in enumerate(elements, start=1):
        component = elem.get("component", "")
        function = elem.get("function", "")
        lines.append(f"{idx}. {component} — {function}")
    return "\n".join(lines)


def sanitize_query_string(raw_query: str) -> str:
    """
    Clean a generated search query by stripping leading/trailing whitespace
    and removing any stray newlines or tab characters.

    Args:
        raw_query: A single search query string as generated by the LLM.

    Returns:
        A clean, single-line search query string.
    """
    return re.sub(r"[\n\r\t]+", " ", raw_query).strip()


def validate_and_pad_query_list(
    queries: list[str],
    required_count: int,
    fallback_prefix: str,
) -> list[str]:
    """
    Ensure a query list has exactly the required number of entries.

    If the LLM returns fewer queries than expected, pad with fallback placeholders.
    If more are returned, truncate to the required count.

    Args:
        queries: List of query strings from LLM output.
        required_count: The exact count expected.
        fallback_prefix: Prefix for any auto-generated fallback query strings.

    Returns:
        A list of exactly `required_count` clean query strings.
    """
    sanitized = [sanitize_query_string(q) for q in queries if q.strip()]
    while len(sanitized) < required_count:
        sanitized.append(f"{fallback_prefix} {len(sanitized) + 1}")
    return sanitized[:required_count]


# =============================================================================
# MAIN QUERY GENERATION FUNCTION
# =============================================================================


def generate_infringement_search_queries(
    industry: str,
    domain: str,
    subdomain: str,
    claim_elements: list[dict[str, str]],
    novelty_summary: str,
) -> dict[str, list[str]]:
    """
    Use LLM to generate two sets of Google search queries for patent infringement detection.

    Set A contains COMPANY_QUERY_COUNT company discovery queries targeting US companies
    in the patent's domain. Set B contains PRODUCT_QUERY_COUNT product discovery queries
    targeting specific commercial products implementing the claimed technology.

    Args:
        industry: The high-level industry classification (e.g., 'Automotive').
        domain: The technology domain (e.g., 'Automotive Electronics').
        subdomain: The sub-domain classification (e.g., 'Vehicle Lighting Control').
        claim_elements: List of dicts with 'component' and 'function' keys.
        novelty_summary: 2-3 sentence description of the patent's technical novelty.

    Returns:
        Dict with keys 'company_queries' (list of 4 strings) and
        'product_queries' (list of 6 strings).
    """
    elements_list_text = format_claim_elements_for_prompt(claim_elements)

    user_prompt = _QUERY_BUILDER_USER_PROMPT.format(
        industry=industry,
        domain=domain,
        subdomain=subdomain,
        elements_list=elements_list_text,
        novelty_summary=novelty_summary[:500],
        scope=ANALYSIS_SCOPE,
        company_query_count=COMPANY_QUERY_COUNT,
        product_query_count=PRODUCT_QUERY_COUNT,
    )

    try:
        result = _call_llm_for_json(_QUERY_BUILDER_SYSTEM_PROMPT, user_prompt)

        raw_company_queries = result.get("company_queries", [])
        raw_product_queries = result.get("product_queries", [])

        company_queries = validate_and_pad_query_list(
            raw_company_queries,
            COMPANY_QUERY_COUNT,
            f"US {domain} companies technology",
        )
        product_queries = validate_and_pad_query_list(
            raw_product_queries,
            PRODUCT_QUERY_COUNT,
            f"US {domain} product {subdomain}",
        )

        logger.info(
            "Generated %d company queries and %d product queries",
            len(company_queries),
            len(product_queries),
        )

        return {
            "company_queries": company_queries,
            "product_queries": product_queries,
        }

    except Exception as exc:
        logger.error("Failed to generate search queries: %s", exc)
        # Return deterministic fallback queries if LLM fails
        fallback_base = f"{industry} {domain} {subdomain} United States"
        return {
            "company_queries": [
                f"{fallback_base} company technology",
                f"US {domain} manufacturer site:com",
                f"{subdomain} company United States OR USA",
                f"{industry} technology company {subdomain} product",
            ],
            "product_queries": [
                f"{subdomain} product United States",
                f"{domain} product specification site:com",
                f"{industry} {subdomain} commercial product US",
                f"{subdomain} technology product announcement",
                f"{domain} commercial implementation US company",
                f"{industry} {subdomain} product release",
            ],
        }
