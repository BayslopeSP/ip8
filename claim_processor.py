"""
claim_processor.py — IP8 Patent Infringement Detection System
Analyzes patent claims using LLM: classifies independent vs. dependent claims,
assesses novelty and domain, and decomposes selected claims into functional elements.
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
)

logger = logging.getLogger(__name__)

# =============================================================================
# LLM CLIENT INITIALIZATION
# =============================================================================

_openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """
    Return a singleton OpenAI client instance, initializing it on first call.

    Returns:
        Configured OpenAI client.
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# =============================================================================
# PROMPTS
# =============================================================================

_PATENT_ATTORNEY_SYSTEM_PROMPT = """You are a senior USPTO patent attorney and technology analyst \
with 20 years of experience in patent claim interpretation, novelty assessment, \
and technology classification. You respond with precise, structured analysis. \
Always return valid JSON when asked. Do not include markdown code fences or extra explanation."""

_NOVELTY_USER_PROMPT = """Analyze the following patent and identify what is technically novel \
about it compared to prior art. Describe the novelty in exactly 2-3 sentences, focusing on \
the specific technical contribution that distinguishes this patent from existing technology.

Patent Title: {title}
Abstract: {abstract}
Claims: {claims_text}

Return JSON:
{{
  "novelty_summary": "<2-3 sentence novelty description>",
  "background_summary": "<2-3 sentence background / problem the patent solves>"
}}"""

_DOMAIN_CLASSIFICATION_USER_PROMPT = """Classify the following patent into its technology domain.

Patent Title: {title}
Abstract: {abstract}
CPC Codes: {cpc_codes_text}

Return EXACTLY this JSON structure with these 3 fields:
{{
  "domain": "<e.g., Automotive Electronics>",
  "subdomain": "<e.g., Vehicle Lighting Control Systems>",
  "industry": "<e.g., Automotive>"
}}"""

_CLAIM_CLASSIFICATION_USER_PROMPT = """Classify the claims in the following patent into \
independent and dependent claims.

An INDEPENDENT claim stands alone and does not reference another claim.
A DEPENDENT claim references ('the system of claim X' or 'the method of claim X') another claim.

Breadth rule:
- BROAD = fewer structural limitations, describes the invention at a high conceptual level
- NARROW = adds specific constraints, sub-components, or conditions to another claim

Claims text:
{claims_text}

Return EXACTLY this JSON structure:
{{
  "independent_claims": [
    {{
      "claim_number": 1,
      "text": "<full claim text>",
      "breadth": "BROAD"
    }}
  ],
  "dependent_claims": [
    {{
      "claim_number": 2,
      "text": "<full claim text>",
      "references_claim": 1
    }}
  ]
}}"""

_CLAIM_DECOMPOSITION_SYSTEM_PROMPT = """You are a patent claims analyst. Your task is to \
decompose a patent claim into its distinct functional elements. Each element must describe a \
single structural or functional component of the invention and its specific role. Be precise \
and technical. Do not merge multiple features into one element. \
Always return valid JSON. Do not include markdown code fences."""

_CLAIM_DECOMPOSITION_USER_PROMPT = """Decompose the following patent claim into EXACTLY its \
core functional elements (aim for 4-7 elements).
For each element, describe:
1. The component (what it is)
2. Its function in the context of the claim (what it does)

Return JSON:
{{
  "elements": [
    {{
      "component": "<component name, e.g., optical sensor>",
      "function": "<what it does, e.g., detects ambient light levels to trigger automatic switching between beam modes>"
    }}
  ]
}}

Claim: {claim_text}"""


# =============================================================================
# HELPERS
# =============================================================================


def _call_llm_for_json(system_prompt: str, user_prompt: str) -> dict:
    """
    Invoke the LLM with given prompts and parse the response as JSON.

    Args:
        system_prompt: The system-level instruction for the LLM.
        user_prompt: The user-level query/context prompt.

    Returns:
        Parsed JSON dict from the LLM response.

    Raises:
        json.JSONDecodeError: If the response cannot be parsed as JSON.
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
    # Strip markdown code fences if present
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)
    return json.loads(raw_text)


def truncate_text_for_prompt(text: str, max_chars: int = 3000) -> str:
    """
    Truncate a text string to fit within LLM prompt length constraints.

    Args:
        text: The full text to potentially truncate.
        max_chars: Maximum number of characters to include.

    Returns:
        Possibly truncated text with ellipsis if cut short.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def format_cpc_codes_as_text(cpc_codes: list[dict[str, str]]) -> str:
    """
    Convert a list of CPC code dicts to a plain text string for prompt injection.

    Args:
        cpc_codes: List of dicts with 'code' and 'label' keys.

    Returns:
        Comma-separated string of 'CODE: label' pairs.
    """
    if not cpc_codes:
        return "Not available"
    parts = []
    for entry in cpc_codes:
        code = entry.get("code", "")
        label = entry.get("label", "")
        parts.append(f"{code}: {label}" if label else code)
    return "; ".join(parts)


# =============================================================================
# MAIN LLM ANALYSIS FUNCTIONS
# =============================================================================


def extract_patent_novelty_and_background(
    title: str,
    abstract: str,
    claims_text: str,
) -> dict[str, str]:
    """
    Use LLM to summarize the technical novelty and background of a patent.

    Args:
        title: The patent title.
        abstract: The patent abstract text.
        claims_text: The full text of all patent claims.

    Returns:
        Dict with keys 'novelty_summary' and 'background_summary', each
        containing a 2-3 sentence description.
    """
    user_prompt = _NOVELTY_USER_PROMPT.format(
        title=title,
        abstract=truncate_text_for_prompt(abstract, 1500),
        claims_text=truncate_text_for_prompt(claims_text, 2000),
    )
    try:
        result = _call_llm_for_json(
            _PATENT_ATTORNEY_SYSTEM_PROMPT, user_prompt)
        return {
            "novelty_summary": result.get("novelty_summary", "Not available"),
            "background_summary": result.get("background_summary", "Not available"),
        }
    except Exception as exc:
        logger.error("Failed to extract novelty/background: %s", exc)
        return {
            "novelty_summary": "Analysis unavailable.",
            "background_summary": "Analysis unavailable.",
        }


def classify_patent_domain_and_industry(
    title: str,
    abstract: str,
    cpc_codes: list[dict[str, str]],
) -> dict[str, str]:
    """
    Use LLM to classify the patent into domain, subdomain, and industry.

    Args:
        title: The patent title.
        abstract: The patent abstract text.
        cpc_codes: List of CPC code dicts.

    Returns:
        Dict with keys 'domain', 'subdomain', and 'industry'.
    """
    cpc_text = format_cpc_codes_as_text(cpc_codes)
    user_prompt = _DOMAIN_CLASSIFICATION_USER_PROMPT.format(
        title=title,
        abstract=truncate_text_for_prompt(abstract, 1500),
        cpc_codes_text=cpc_text,
    )
    try:
        result = _call_llm_for_json(
            _PATENT_ATTORNEY_SYSTEM_PROMPT, user_prompt)
        return {
            "domain": result.get("domain", "Technology"),
            "subdomain": result.get("subdomain", "General"),
            "industry": result.get("industry", "Technology"),
        }
    except Exception as exc:
        logger.error("Failed to classify domain/industry: %s", exc)
        return {
            "domain": "Technology",
            "subdomain": "General",
            "industry": "Technology",
        }


def classify_patent_claims(claims_text: str) -> dict[str, list]:
    """
    Use LLM to classify all patent claims into independent and dependent buckets.

    Each independent claim includes a breadth label (BROAD or NARROW).
    Each dependent claim includes a reference to the parent claim number.

    Args:
        claims_text: The full text of all patent claims.

    Returns:
        Dict with 'independent_claims' and 'dependent_claims' lists.
    """
    user_prompt = _CLAIM_CLASSIFICATION_USER_PROMPT.format(
        claims_text=truncate_text_for_prompt(claims_text, 4000),
    )
    try:
        result = _call_llm_for_json(
            _PATENT_ATTORNEY_SYSTEM_PROMPT, user_prompt)
        independent_claims = result.get("independent_claims", [])
        dependent_claims = result.get("dependent_claims", [])
        return {
            "independent_claims": independent_claims,
            "dependent_claims": dependent_claims,
        }
    except Exception as exc:
        logger.error("Failed to classify claims: %s", exc)
        return {"independent_claims": [], "dependent_claims": []}


def decompose_claim_into_elements(claim_text: str) -> list[dict[str, str]]:
    """
    Use LLM to decompose a single patent claim into functional elements.

    Each element identifies a component and its specific function within the claim.
    The LLM is instructed to return between 4 and 7 elements.

    Args:
        claim_text: The full text of the target patent claim.

    Returns:
        List of dicts, each with 'component' and 'function' keys.
    """
    user_prompt = _CLAIM_DECOMPOSITION_USER_PROMPT.format(
        claim_text=truncate_text_for_prompt(claim_text, 2000),
    )
    try:
        result = _call_llm_for_json(
            _CLAIM_DECOMPOSITION_SYSTEM_PROMPT, user_prompt)
        elements = result.get("elements", [])
        return [
            {
                "component": elem.get("component", ""),
                "function": elem.get("function", ""),
            }
            for elem in elements
            if elem.get("component")
        ]
    except Exception as exc:
        logger.error("Failed to decompose claim into elements: %s", exc)
        return []


def run_full_claim_preprocessing(
    patent_data,
) -> dict:
    """
    Execute all LLM preprocessing steps for a patent and return a unified results dict.

    Calls novelty extraction, domain classification, and claim classification LLM
    functions sequentially, then assembles the results into a single dictionary
    suitable for caching and display.

    Args:
        patent_data: A PatentData instance with title, abstract, claims, and CPC data.

    Returns:
        Dict containing keys: novelty_summary, background_summary, domain, subdomain,
        industry, independent_claims, dependent_claims.
    """
    logger.info("Running full claim preprocessing for patent: %s",
                patent_data.patent_number)

    novelty_result = extract_patent_novelty_and_background(
        title=patent_data.title,
        abstract=patent_data.abstract,
        claims_text=patent_data.full_claims_text,
    )

    domain_result = classify_patent_domain_and_industry(
        title=patent_data.title,
        abstract=patent_data.abstract,
        cpc_codes=patent_data.cpc_codes,
    )

    claims_result = classify_patent_claims(
        claims_text=patent_data.full_claims_text)

    return {
        "novelty_summary": novelty_result["novelty_summary"],
        "background_summary": novelty_result["background_summary"],
        "domain": domain_result["domain"],
        "subdomain": domain_result["subdomain"],
        "industry": domain_result["industry"],
        "independent_claims": claims_result["independent_claims"],
        "dependent_claims": claims_result["dependent_claims"],
    }
