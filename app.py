"""
app.py — IP8 Patent Infringement Detection System
Streamlit entry point. Orchestrates the full patent infringement detection pipeline
through a multi-step UI: fetch → preprocess → select claims → decompose elements
→ generate queries → search → scrape → analyze → display results.

Run with:
    streamlit run app.py
"""

import logging
import traceback

import streamlit as st

import config
from cache_manager import (
    cache_exists,
    clear_cache_for_patent,
    get_cache_size_bytes,
    load_from_cache,
    save_to_cache,
)
from patent_fetcher import fetch_patent_details
from claim_processor import (
    decompose_claim_into_elements,
    run_full_claim_preprocessing,
)
from query_builder import generate_infringement_search_queries
from searcher import search_and_collect_candidate_urls
from scraper import create_headless_chrome_driver, scrape_page_content, ScrapedPage
from analyzer import analyze_all_scraped_pages
from ui_components import (
    render_step_header,
    render_patent_summary_card,
    render_cpc_codes_table,
    render_independent_claims_checkboxes,
    render_dependent_claims_table,
    render_claim_elements_checklist,
    render_generated_queries,
    render_full_results_dashboard,
    render_csv_export_button,
    render_pipeline_progress,
    render_cache_info_banner,
)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.APP_LAYOUT,
    initial_sidebar_state="expanded",
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state() -> None:
    """
    Initialize all required Streamlit session state keys to their default values.

    This must be called once at the top of every Streamlit render cycle to ensure
    that all downstream code can safely read from session state without KeyErrors.
    """
    defaults = {
        "patent_number": "",
        "patent_data": None,
        "preprocessing_results": None,
        "selected_claim_numbers": [],
        "confirmed_claims": False,
        "claim_elements_map": {},        # {claim_number: [element dicts]}
        # {claim_number: [selected element dicts]}
        "selected_elements_map": {},
        "confirmed_elements": False,
        "industry_scope_all": False,
        "queries": None,
        "queries_confirmed": False,
        "candidate_urls": None,
        "scraped_pages": None,
        "analysis_results": None,
        "pipeline_complete": False,
        "pipeline_running": False,
        "loaded_from_cache": False,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# =============================================================================
# CACHE CONVENIENCE WRAPPERS
# =============================================================================

_CACHE_KEY_PATENT = "patent_data"
_CACHE_KEY_PREPROCESSING = "preprocessing"
_CACHE_KEY_QUERIES = "queries"
_CACHE_KEY_SCRAPED = "scraped_pages"
_CACHE_KEY_RESULTS = "analysis_results"


def attempt_load_full_pipeline_from_cache(patent_number: str) -> bool:
    """
    Try to load all pipeline steps for a patent from disk cache into session state.

    Loads patent data, preprocessing results, queries, scraped pages, and analysis
    results if all cache files are present. Returns True only if every step was loaded.

    Args:
        patent_number: The patent identifier to look up in cache.

    Returns:
        True if all steps were loaded successfully, False otherwise.
    """
    steps = [
        (_CACHE_KEY_PATENT, "patent_data"),
        (_CACHE_KEY_PREPROCESSING, "preprocessing_results"),
        (_CACHE_KEY_QUERIES, "queries"),
        (_CACHE_KEY_SCRAPED, "scraped_pages"),
        (_CACHE_KEY_RESULTS, "analysis_results"),
    ]
    for cache_key, session_key in steps:
        if not cache_exists(patent_number, cache_key):
            return False
        data = load_from_cache(patent_number, cache_key)
        if data is None:
            return False
        st.session_state[session_key] = data

    # Restore derived state for UI rendering
    preprocessing = st.session_state["preprocessing_results"]
    if preprocessing:
        independent_claims = preprocessing.get("independent_claims", [])
        if len(independent_claims) == 1:
            st.session_state["selected_claim_numbers"] = [
                independent_claims[0].get("claim_number", 1)
            ]
        elif independent_claims:
            st.session_state["selected_claim_numbers"] = [
                c.get("claim_number") for c in independent_claims
            ]

    st.session_state["confirmed_claims"] = True
    st.session_state["confirmed_elements"] = True
    st.session_state["pipeline_complete"] = True
    st.session_state["loaded_from_cache"] = True
    return True


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> str:
    """
    Render the application sidebar containing patent input, cache controls,
    and masked API key display.

    Returns:
        The patent number string entered by the user (may be empty).
    """
    with st.sidebar:
        st.title("🔍 IP8")
        st.caption("Patent Infringement Detection System")
        st.markdown("---")

        patent_input = st.text_input(
            "Patent Number",
            placeholder="e.g., US10696212B2",
            help="Enter a US patent number in standard format.",
            key="sidebar_patent_input",
        ).strip().upper()

        st.markdown("---")
        st.markdown("**Cache Controls**")

        if patent_input:
            cache_size = get_cache_size_bytes(patent_input)
            if cache_size > 0:
                size_kb = cache_size / 1024
                st.caption(f"📦 Cache: {size_kb:.1f} KB for {patent_input}")
                if st.button("🗑️ Clear Cache & Re-run", use_container_width=True):
                    deleted = clear_cache_for_patent(patent_input)
                    st.success(
                        f"Deleted {deleted} cache file(s). Re-running analysis.")
                    _reset_pipeline_session_state()
                    st.rerun()
            else:
                st.caption("No cache found for this patent.")

        st.markdown("---")
        st.markdown("**API Configuration**")
        masked_openai = (
            config.OPENAI_API_KEY[:8] + "..." + config.OPENAI_API_KEY[-4:]
            if len(config.OPENAI_API_KEY) > 12
            else "Not configured"
        )
        masked_serp = (
            config.SERP_API_KEY[:8] + "..." + config.SERP_API_KEY[-4:]
            if len(config.SERP_API_KEY) > 12
            else "Not configured"
        )
        st.caption(f"OpenAI: `{masked_openai}`")
        st.caption(f"SerpAPI: `{masked_serp}`")

        st.markdown("---")
        st.caption(f"LLM: `{config.LLM_MODEL}` | T=0 | Seed=42")
        st.caption(f"Scope: 🇺🇸 {config.ANALYSIS_SCOPE} only")

    return patent_input


# =============================================================================
# PIPELINE RESET
# =============================================================================

def _reset_pipeline_session_state() -> None:
    """
    Reset all pipeline-related session state keys to their default values.

    Called when the user changes the patent number or clears the cache to ensure
    a clean slate for the new analysis run.
    """
    pipeline_keys = [
        "patent_data", "preprocessing_results", "selected_claim_numbers",
        "confirmed_claims", "claim_elements_map", "selected_elements_map",
        "confirmed_elements", "industry_scope_all", "queries",
        "queries_confirmed", "candidate_urls", "scraped_pages",
        "analysis_results", "pipeline_complete", "pipeline_running",
        "loaded_from_cache",
    ]
    for key in pipeline_keys:
        if key in st.session_state:
            del st.session_state[key]
    initialize_session_state()


# =============================================================================
# STEP 2: PATENT FETCH
# =============================================================================

def run_step_patent_fetch(patent_number: str) -> bool:
    """
    Execute Step 2: fetch patent details from Google Patents and cache the result.

    Stores the resulting PatentData object in session state and disk cache.
    Displays progress feedback and any errors via Streamlit.

    Args:
        patent_number: Normalized patent identifier to fetch.

    Returns:
        True if the fetch succeeded, False on error.
    """
    render_pipeline_progress("Fetching patent bibliographic data...", 0.1)

    try:
        patent_data = fetch_patent_details(patent_number)
        st.session_state["patent_data"] = patent_data
        save_to_cache(patent_number, _CACHE_KEY_PATENT, patent_data)

        st.success(
            f"✅ Patent data fetched: **{patent_data.title or patent_number}**")
        return True
    except Exception as exc:
        st.error(f"❌ Failed to fetch patent data: {exc}")
        logger.error("Patent fetch error: %s\n%s", exc, traceback.format_exc())
        return False


# =============================================================================
# STEP 3: CLAIM PREPROCESSING
# =============================================================================

def run_step_claim_preprocessing(patent_number: str) -> bool:
    """
    Execute Step 3: run LLM preprocessing on patent claims and cache results.

    Runs novelty/background extraction, domain classification, and claim
    classification via GPT. Stores results in session state and disk cache.

    Args:
        patent_number: Patent identifier used as cache key.

    Returns:
        True if preprocessing succeeded, False on error.
    """
    render_step_header(2, "Patent Analysis")
    render_pipeline_progress("Running LLM claim analysis...", 0.25)

    patent_data = st.session_state["patent_data"]
    try:
        preprocessing_results = run_full_claim_preprocessing(patent_data)
        st.session_state["preprocessing_results"] = preprocessing_results
        save_to_cache(patent_number, _CACHE_KEY_PREPROCESSING,
                      preprocessing_results)
        st.success("✅ LLM preprocessing complete.")
        return True
    except Exception as exc:
        st.error(f"❌ LLM preprocessing failed: {exc}")
        logger.error("Preprocessing error: %s\n%s",
                     exc, traceback.format_exc())
        return False


def display_preprocessing_results() -> None:
    """
    Display the patent preprocessing results in a styled layout.

    Shows the patent summary card, novelty/background summaries, CPC codes table,
    independent/dependent claims breakdown. Called after preprocessing is complete.
    """
    patent_data = st.session_state.get("patent_data")
    preprocessing = st.session_state.get("preprocessing_results")
    if not patent_data or not preprocessing:
        return

    render_step_header(
        2, "Patent Analysis", completed=True)

    # Patent metadata table
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Patent Metadata**")
        meta_data = {
            "Field": ["Priority Date", "Filing Date", "Assignee", "Inventors", "Domain", "Sub-domain", "Industry"],
            "Value": [
                patent_data.priority_date or "Unknown",
                patent_data.filing_date or "Unknown",
                patent_data.assignee or "Unknown",
                ", ".join(
                    patent_data.inventors) if patent_data.inventors else "Unknown",
                preprocessing.get("domain", ""),
                preprocessing.get("subdomain", ""),
                preprocessing.get("industry", ""),
            ],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(meta_data),
                     use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**CPC Classification Codes**")
        render_cpc_codes_table(patent_data.cpc_codes)

    # Novelty and Background
    st.markdown("**Patent Novelty**")
    st.info(preprocessing.get("novelty_summary", "Not available"))
    st.markdown("**Background / Problem Solved**")
    st.info(preprocessing.get("background_summary", "Not available"))

    # Claims breakdown
    independent_claims = preprocessing.get("independent_claims", [])
    dependent_claims = preprocessing.get("dependent_claims", [])

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            f"**Independent Claims ({len(independent_claims)} found)**")
        if independent_claims:
            for claim in independent_claims:
                st.markdown(
                    f"- **Claim {claim.get('claim_number')}** "
                    f"({claim.get('breadth', '')}) — "
                    f"{claim.get('text', '')[:150]}..."
                )
    with col4:
        st.markdown(f"**Dependent Claims ({len(dependent_claims)} found)**")
        with st.expander("View dependent claims", expanded=False):
            render_dependent_claims_table(dependent_claims)


# =============================================================================
# STEP 3b: CLAIM SELECTION
# =============================================================================

def run_step_claim_selection() -> None:
    """
    Execute Step 3b: render claim selection UI and capture the user's choice.

    Auto-selects if only one independent claim exists. Otherwise shows checkboxes
    for the user to pick which claims to analyze, guarded by a confirm button.
    """
    render_step_header(3, "Claim Selection", completed=st.session_state.get(
        "confirmed_claims", False))

    preprocessing = st.session_state.get("preprocessing_results", {})
    independent_claims = preprocessing.get("independent_claims", [])

    if not independent_claims:
        st.warning(
            "No independent claims were detected. Check patent claims text.")
        return

    if len(independent_claims) == 1:
        claim_number = independent_claims[0].get("claim_number", 1)
        st.session_state["selected_claim_numbers"] = [claim_number]
        st.session_state["confirmed_claims"] = True
        st.success(
            f"✅ Auto-selected the only independent claim: **Claim {claim_number}**")
        return

    # Multiple independent claims — show selection UI
    if not st.session_state.get("confirmed_claims"):
        selected = render_independent_claims_checkboxes(independent_claims)
        if st.button("✅ Confirm Selection & Proceed", key="confirm_claims_btn"):
            if not selected:
                st.error("Please select at least one claim to continue.")
            else:
                st.session_state["selected_claim_numbers"] = selected
                st.session_state["confirmed_claims"] = True
                st.rerun()
    else:
        selected_nums = st.session_state.get("selected_claim_numbers", [])
        st.success(
            f"✅ Selected claims: {', '.join(f'Claim {n}' for n in selected_nums)}")


# =============================================================================
# STEP 4: CLAIM ELEMENT DECOMPOSITION
# =============================================================================

def run_step_claim_decomposition(patent_number: str) -> None:
    """
    Execute Step 4: decompose each selected claim into functional elements using LLM.

    Calls decompose_claim_into_elements() for each selected claim, caches the results,
    and renders an interactive element checklist plus industry scope selection.

    Args:
        patent_number: Patent identifier used as cache key.
    """
    render_step_header(
        4,
        "Claim Element Decomposition",
        completed=st.session_state.get("confirmed_elements", False),
    )

    preprocessing = st.session_state.get("preprocessing_results", {})
    independent_claims = preprocessing.get("independent_claims", [])
    selected_claim_numbers = st.session_state.get("selected_claim_numbers", [])

    if not st.session_state.get("claim_elements_map"):
        # Build a map: claim_number -> list of element dicts
        with st.spinner("Decomposing claim elements via LLM..."):
            elements_map: dict[int, list] = {}
            for claim in independent_claims:
                claim_number = claim.get("claim_number")
                if claim_number not in selected_claim_numbers:
                    continue
                claim_text = claim.get("text", "")
                elements = decompose_claim_into_elements(claim_text)
                elements_map[claim_number] = elements
                logger.info(
                    "Decomposed claim %d into %d elements.", claim_number, len(
                        elements)
                )
            st.session_state["claim_elements_map"] = elements_map

    claim_elements_map = st.session_state.get("claim_elements_map", {})

    if st.session_state.get("confirmed_elements"):
        for claim_number, elements in claim_elements_map.items():
            st.markdown(
                f"**Claim {claim_number} — {len(elements)} element(s):**")
            selected_elements = st.session_state["selected_elements_map"].get(
                claim_number, elements)
            for elem in selected_elements:
                st.markdown(
                    f"  ✅ **{elem.get('component')}** — {elem.get('function')}")
        industry = preprocessing.get("industry", "Technology")
        scope_label = "All industries" if st.session_state.get(
            "industry_scope_all") else f"{industry} only"
        st.success(f"✅ Elements confirmed | Scope: {scope_label}")
        return

    # Interactive checklist for each selected claim
    selected_elements_map: dict[int, list] = {}
    for claim_number, elements in claim_elements_map.items():
        st.markdown(f"**Claim {claim_number} elements:**")
        selected = render_claim_elements_checklist(
            elements,
            session_key_prefix=f"elem_sel_c{claim_number}",
        )
        selected_elements_map[claim_number] = selected

    # Industry scope radio
    st.markdown("---")
    industry = preprocessing.get("industry", "Technology")
    scope_choice = st.radio(
        f"Proceed with **{industry}** only, or analyze across all industries?",
        options=[f"{industry} only", "All industries"],
        key="industry_scope_radio",
    )
    industry_scope_all = scope_choice == "All industries"

    if st.button("✅ Confirm Elements & Build Queries", key="confirm_elements_btn"):
        any_elements_selected = any(
            len(v) > 0 for v in selected_elements_map.values())
        if not any_elements_selected:
            st.error("Please keep at least one element selected.")
        else:
            st.session_state["selected_elements_map"] = selected_elements_map
            st.session_state["industry_scope_all"] = industry_scope_all
            st.session_state["confirmed_elements"] = True
            st.rerun()


# =============================================================================
# STEP 5: QUERY GENERATION
# =============================================================================

def run_step_query_generation(patent_number: str) -> bool:
    """
    Execute Step 5: generate SerpAPI search queries from claim elements using LLM.

    Aggregates all selected elements across all selected claims, then calls
    generate_infringement_search_queries(). After displaying the queries, waits
    for explicit user confirmation before returning True so that Step 6 (search)
    does not start automatically.

    Args:
        patent_number: Patent identifier used as cache key.

    Returns:
        True only after queries are generated AND the user has clicked
        'Proceed to Search'. False in all other cases.
    """
    queries_confirmed = st.session_state.get("queries_confirmed", False)
    queries_exist = st.session_state.get("queries") is not None

    render_step_header(5, "Query Generation", completed=queries_confirmed)

    # Already confirmed — just show a summary and let the pipeline continue.
    if queries_confirmed and queries_exist:
        queries = st.session_state["queries"]
        render_generated_queries(
            queries["company_queries"], queries["product_queries"])
        return True

    preprocessing = st.session_state.get("preprocessing_results", {})
    selected_elements_map = st.session_state.get("selected_elements_map", {})

    # Generate queries if not yet done.
    if not queries_exist:
        # Flatten all selected elements across all claims for the prompt.
        all_selected_elements: list[dict] = []
        for elements in selected_elements_map.values():
            all_selected_elements.extend(elements)

        render_pipeline_progress("Generating search queries via LLM...", 0.40)
        try:
            queries = generate_infringement_search_queries(
                industry=preprocessing.get("industry", "Technology"),
                domain=preprocessing.get("domain", "Technology"),
                subdomain=preprocessing.get("subdomain", "General"),
                claim_elements=all_selected_elements,
                novelty_summary=preprocessing.get("novelty_summary", ""),
            )
            st.session_state["queries"] = queries
            save_to_cache(patent_number, _CACHE_KEY_QUERIES, queries)
            st.success(
                f"✅ Generated {len(queries['company_queries'])} company queries "
                f"and {len(queries['product_queries'])} product queries."
            )
        except Exception as exc:
            st.error(f"❌ Query generation failed: {exc}")
            logger.error("Query generation error: %s\n%s",
                         exc, traceback.format_exc())
            return False

    # Show the queries so the user can review them.
    queries = st.session_state["queries"]
    render_generated_queries(
        queries["company_queries"], queries["product_queries"])

    # User must explicitly approve before Step 6 begins.
    st.info("Review the generated queries above, then click the button below to start searching.")
    if st.button("🔍 Proceed to Search", key="confirm_queries_btn", type="primary"):
        st.session_state["queries_confirmed"] = True
        st.rerun()

    return False


# =============================================================================
# STEP 6: SEARCH
# =============================================================================

def run_step_serpapi_search(patent_number: str) -> bool:
    """
    Execute Step 6: run SerpAPI searches and collect candidate URLs.

    Calls search_and_collect_candidate_urls() with the generated queries.
    Results are stored in session state (not separately cached since they feed
    directly into the scraper which is cached).

    Args:
        patent_number: Patent identifier (used for logging context).

    Returns:
        True if search yielded at least one URL, False on error or empty result.
    """
    render_step_header(6, "SerpAPI Search")

    # --- Key validation before hitting the API ---
    if not config.SERP_API_KEY:
        st.error(
            "❌ **SERP_API_KEY is not set.**  "
            "Open the `.env` file and add `SERP_API_KEY=your_key_here`, then restart the app."
        )
        return False

    render_pipeline_progress("Executing Google searches via SerpAPI...", 0.50)

    queries = st.session_state.get("queries", {})
    try:
        candidate_urls = search_and_collect_candidate_urls(
            company_queries=queries.get("company_queries", []),
            product_queries=queries.get("product_queries", []),
        )
        st.session_state["candidate_urls"] = candidate_urls
        if not candidate_urls:
            st.warning(
                "⚠️ No candidate URLs found after searching and filtering.  "
                "This usually means all results were blacklisted or SerpAPI returned "
                "empty pages for these queries. Check your SERP_API_KEY quota at "
                "https://serpapi.com/dashboard and try re-running."
            )
            return False
        st.success(
            f"✅ Found {len(candidate_urls)} candidate URLs for scraping.")
        return True
    except RuntimeError as exc:
        # SerpAPI returned an explicit error (bad key, quota exceeded, etc.)
        error_text = str(exc)
        st.error(f"❌ SerpAPI API error: **{error_text}**")
        if "Invalid API key" in error_text or "api_key" in error_text.lower():
            st.info(
                "Your SERP_API_KEY appears to be invalid. "
                "Check the key at https://serpapi.com/dashboard and update your `.env` file."
            )
        elif "credit" in error_text.lower() or "quota" in error_text.lower() or "plan" in error_text.lower():
            st.info(
                "Your SerpAPI account has run out of credits or searches. "
                "Check your plan at https://serpapi.com/dashboard."
            )
        logger.error("Search error: %s\n%s", exc, traceback.format_exc())
        return False
    except Exception as exc:
        st.error(f"❌ SerpAPI search failed: {exc}")
        logger.error("Search error: %s\n%s", exc, traceback.format_exc())
        return False


# =============================================================================
# STEP 7: SCRAPING
# =============================================================================

def run_step_web_scraping(patent_number: str) -> bool:
    """
    Execute Step 7: scrape all candidate URLs with headless Selenium Chrome.

    Calls scrape_all_candidate_urls() and stores the list of ScrapedPage objects
    in session state and disk cache.

    Args:
        patent_number: Patent identifier used as cache key.

    Returns:
        True if at least one page was successfully scraped, False on error.
    """
    render_step_header(7, "Web Scraping")
    candidate_urls = st.session_state.get("candidate_urls", [])

    if st.session_state.get("scraped_pages"):
        scraped_pages = st.session_state["scraped_pages"]
        successful = sum(1 for p in scraped_pages if p.scrape_success)
        st.success(
            f"✅ Scraped {successful}/{len(scraped_pages)} pages (from cache).")
        return True

    progress_placeholder = st.empty()
    scraped_pages: list[ScrapedPage] = []
    total = len(candidate_urls)

    driver = None
    try:
        driver = create_headless_chrome_driver()
        for idx, url in enumerate(candidate_urls, start=1):
            with progress_placeholder.container():
                st.progress((idx - 1) / total)
                st.markdown(
                    f"**Scraping page {idx}/{total}\u2026** `{url[:80]}`")
            page = scrape_page_content(url, driver)
            scraped_pages.append(page)
            with progress_placeholder.container():
                st.progress(idx / total)
                st.markdown(f"**Scraping {idx}/{total} pages complete**")
                with st.expander(
                    f"Pages scraped so far ({idx}/{total})", expanded=True
                ):
                    for p in scraped_pages:
                        icon = "✅" if p.scrape_success else "❌"
                        chars = (
                            f"{len(p.page_text):,} chars"
                            if p.scrape_success
                            else p.error_message[:60] or "Failed"
                        )
                        date_str = (
                            f" | Date: {p.extracted_date}"
                            if p.extracted_date
                            else ""
                        )
                        st.markdown(
                            f"{icon} `{p.url[:80]}` — {chars}{date_str}")
    except Exception as exc:
        st.error(f"❌ Scraping failed: {exc}")
        logger.error("Scraping error: %s\n%s", exc, traceback.format_exc())
        return False
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    st.session_state["scraped_pages"] = scraped_pages
    save_to_cache(patent_number, _CACHE_KEY_SCRAPED, scraped_pages)
    progress_placeholder.empty()

    successful = sum(1 for p in scraped_pages if p.scrape_success)
    st.success(
        f"✅ Scraping complete: {successful}/{len(scraped_pages)} pages successful.")
    return successful > 0


# =============================================================================
# STEPS 8–9: ANALYSIS & SCORING
# =============================================================================


def _render_scoring_summary(results: list) -> None:
    """Show Step 9 scoring header and high-level match metrics."""
    render_step_header(9, "Infringement Scoring", completed=True)
    high_count = sum(
        1 for r in results if r.infringement_label == config.SCORE_LABEL_HIGH)
    medium_count = sum(
        1 for r in results if r.infringement_label == config.SCORE_LABEL_MEDIUM)
    low_count = len(results) - high_count - medium_count
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Matches", len(results))
    sc2.metric("🔴 HIGH Risk", high_count)
    sc3.metric("🟠 MEDIUM Risk", medium_count)
    sc4.metric("🟡 LOW Risk", low_count)


def run_step_infringement_analysis(patent_number: str) -> bool:
    """
    Execute Steps 8 and 9: analyze all scraped pages for infringement and score them.

    Calls analyze_all_scraped_pages() which internally runs LLM analysis per page
    and computes infringement scores. Results are stored in session state and cached.

    Args:
        patent_number: Patent identifier used as cache key.

    Returns:
        True if analysis completed (even with zero results), False on error.
    """
    render_step_header(8, "Product Analysis")

    if st.session_state.get("analysis_results") is not None:
        results = st.session_state["analysis_results"]
        st.success(
            f"✅ Analysis complete: {len(results)} qualifying matches (from cache).")
        if results:
            _render_scoring_summary(results)
        return True

    patent_data = st.session_state.get("patent_data")
    preprocessing = st.session_state.get("preprocessing_results", {})
    scraped_pages = st.session_state.get("scraped_pages", [])
    selected_claim_numbers = st.session_state.get("selected_claim_numbers", [])
    selected_elements_map = st.session_state.get("selected_elements_map", {})

    # Build selected claim text and combined elements list from the first selected claim
    # (multi-claim analysis is iterative; we run the primary claim first)
    independent_claims = preprocessing.get("independent_claims", [])
    selected_claims_data = [
        c for c in independent_claims
        if c.get("claim_number") in selected_claim_numbers
    ]

    if not selected_claims_data:
        st.error(
            "No selected claim data found. Please go back and reconfirm claim selection.")
        return False

    total_pages = len(scraped_pages)
    render_pipeline_progress(
        f"Analyzing {total_pages} pages against {len(selected_claims_data)} claim(s)...",
        0.75,
    )

    try:
        all_qualifying_results = []

        for claim in selected_claims_data:
            claim_number = claim.get("claim_number")
            claim_text = claim.get("text", "")
            claim_elements = selected_elements_map.get(claim_number, [])

            if not claim_elements:
                logger.warning(
                    "No elements for claim %d — skipping.", claim_number)
                continue

            claim_results = analyze_all_scraped_pages(
                scraped_pages=scraped_pages,
                patent_title=patent_data.title if patent_data else "",
                priority_date=patent_data.priority_date if patent_data else "",
                novelty_summary=preprocessing.get("novelty_summary", ""),
                domain=preprocessing.get("domain", ""),
                subdomain=preprocessing.get("subdomain", ""),
                selected_claim_text=claim_text,
                claim_elements=claim_elements,
            )
            all_qualifying_results.extend(claim_results)

        # Deduplicate by URL (keep highest-scoring entry per URL)
        url_to_best_result: dict[str, object] = {}
        for result in all_qualifying_results:
            existing = url_to_best_result.get(result.url)
            if existing is None or result.infringement_score > existing.infringement_score:
                url_to_best_result[result.url] = result

        final_results = sorted(
            url_to_best_result.values(),
            key=lambda r: r.infringement_score,
            reverse=True,
        )

        st.session_state["analysis_results"] = final_results
        save_to_cache(patent_number, _CACHE_KEY_RESULTS, final_results)
        st.session_state["pipeline_complete"] = True

        st.success(
            f"✅ Analysis complete: **{len(final_results)}** qualifying match(es) found.")
        if final_results:
            _render_scoring_summary(final_results)
        return True

    except Exception as exc:
        st.error(f"❌ Infringement analysis failed: {exc}")
        logger.error("Analysis error: %s\n%s", exc, traceback.format_exc())
        return False


# =============================================================================
# MAIN APP
# =============================================================================

def main() -> None:
    """
    Main Streamlit application entry point.

    Renders the sidebar, orchestrates the multi-step pipeline based on session
    state progression, and displays results. Each step gates the next — the user
    must proceed sequentially through the pipeline.
    """
    initialize_session_state()

    # Title & description
    st.title(f"🔍 {config.APP_TITLE}")
    st.caption(
        "Analyzes US patent claims and scores real-world products by infringement likelihood "
        f"using GPT-4o-mini + SerpAPI. Scope: 🇺🇸 {config.ANALYSIS_SCOPE} only."
    )
    st.markdown("---")

    # Render sidebar and get patent input
    patent_number = render_sidebar()

    if not patent_number:
        st.info("👈 Enter a patent number in the sidebar to begin.")
        st.markdown(
            """
            **How it works:**
            1. Enter a US patent number (e.g., `US10696212B2`)
            2. Click **Analyze Patent** to start the pipeline
            3. Review LLM-extracted claims and select which to analyze
            4. Confirm claim elements and scope
            5. View scored infringement results with per-element breakdowns

            Results are cached locally for instant re-runs.
            """
        )
        return

    # Detect patent number change — reset if different from last run
    if st.session_state.get("patent_number") and st.session_state["patent_number"] != patent_number:
        _reset_pipeline_session_state()
    st.session_state["patent_number"] = patent_number

    # ─── STEP 1: Patent Input ─────────────────────────────────────────────────
    render_step_header(1, "Patent Input")
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        st.info(f"Selected patent: **{patent_number}**")
    with col_btn:
        analyze_button_clicked = st.button(
            "🚀 Analyze Patent",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("pipeline_running", False),
        )

    # ─── CACHE CHECK ──────────────────────────────────────────────────────────
    if analyze_button_clicked and not st.session_state.get("patent_data"):
        if attempt_load_full_pipeline_from_cache(patent_number):
            cache_size = get_cache_size_bytes(patent_number)
            render_cache_info_banner(patent_number, cache_size)
        else:
            st.session_state["pipeline_running"] = True

    # ─── DISPLAY RESULTS IF LOADED FROM CACHE ────────────────────────────────
    if st.session_state.get("loaded_from_cache") and st.session_state.get("pipeline_complete"):
        cache_size = get_cache_size_bytes(patent_number)
        render_cache_info_banner(patent_number, cache_size)
        _display_completed_pipeline()
        return

    # ─── PIPELINE EXECUTION ───────────────────────────────────────────────────
    if not st.session_state.get("pipeline_running") and not st.session_state.get("patent_data"):
        if not analyze_button_clicked:
            st.info("Click **Analyze Patent** to start the pipeline.")
        return

    # Step 2: Fetch patent data
    if not st.session_state.get("patent_data"):
        success = run_step_patent_fetch(patent_number)
        if not success:
            st.session_state["pipeline_running"] = False
            return
        st.rerun()

    # Step 3: Preprocessing
    if not st.session_state.get("preprocessing_results"):
        success = run_step_claim_preprocessing(patent_number)
        if not success:
            st.session_state["pipeline_running"] = False
            return
        st.rerun()

    # Show preprocessing results
    display_preprocessing_results()
    st.markdown("---")

    # Step 3b: Claim selection
    run_step_claim_selection()
    if not st.session_state.get("confirmed_claims"):
        return
    st.markdown("---")

    # Step 4: Claim element decomposition
    run_step_claim_decomposition(patent_number)
    if not st.session_state.get("confirmed_elements"):
        return
    st.markdown("---")

    # Step 5: Query generation
    success = run_step_query_generation(patent_number)
    if not success:
        return
    st.markdown("---")

    # Step 6: SerpAPI Search
    if not st.session_state.get("candidate_urls"):
        success = run_step_serpapi_search(patent_number)
        if not success:
            st.session_state["pipeline_running"] = False
            return
        st.rerun()
    else:
        render_step_header(6, "SerpAPI Search", completed=True)
        st.success(
            f"✅ {len(st.session_state['candidate_urls'])} candidate URLs collected."
        )
    st.markdown("---")

    # Step 7: Scraping
    if not st.session_state.get("scraped_pages"):
        success = run_step_web_scraping(patent_number)
        if not success:
            st.session_state["pipeline_running"] = False
            return
        st.rerun()
    else:
        render_step_header(7, "Web Scraping", completed=True)
        scraped = st.session_state["scraped_pages"]
        successful = sum(1 for p in scraped if p.scrape_success)
        st.success(f"✅ {successful}/{len(scraped)} pages scraped.")
    st.markdown("---")

    # Steps 8+9: Analysis
    if st.session_state.get("analysis_results") is None:
        success = run_step_infringement_analysis(patent_number)
        if not success:
            st.session_state["pipeline_running"] = False
            return
    else:
        render_step_header(
            8, "Product Analysis & Infringement Scoring", completed=True)
        st.success(
            f"✅ {len(st.session_state['analysis_results'])} qualifying match(es) found."
        )

    st.session_state["pipeline_running"] = False
    st.markdown("---")

    # Step 10: Results Dashboard
    _display_completed_pipeline()


def _display_completed_pipeline() -> None:
    """
    Render the final results dashboard and CSV export button.

    Called either after a fresh pipeline run or after loading all data from cache.
    Pulls all required data from session state.
    """
    patent_data = st.session_state.get("patent_data")
    preprocessing = st.session_state.get("preprocessing_results", {})
    analysis_results = st.session_state.get("analysis_results", [])
    selected_claim_numbers = st.session_state.get("selected_claim_numbers", [])

    if not patent_data or preprocessing is None:
        st.warning("Pipeline data incomplete. Please re-run the analysis.")
        return

    render_full_results_dashboard(
        analysis_results=analysis_results,
        patent_number=patent_data.patent_number,
        title=patent_data.title,
        priority_date=patent_data.priority_date,
        domain=preprocessing.get("domain", ""),
        industry=preprocessing.get("industry", ""),
        selected_claim_numbers=selected_claim_numbers,
        assignee=patent_data.assignee,
    )

    st.markdown("---")
    render_csv_export_button(analysis_results, patent_data.patent_number)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
