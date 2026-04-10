"""
ui_components.py — IP8 Patent Infringement Detection System
Reusable Streamlit UI components for the patent infringement dashboard.
Provides styled cards, badges, tables, patent summaries, and results panels.
"""

import pandas as pd
import streamlit as st

from config import (
    BADGE_HIGH_COLOR,
    BADGE_MEDIUM_COLOR,
    BADGE_LOW_COLOR,
    SCORE_LABEL_HIGH,
    SCORE_LABEL_MEDIUM,
    SCORE_LABEL_LOW,
)


# =============================================================================
# BADGE & COLOR HELPERS
# =============================================================================


def get_badge_color_for_label(infringement_label: str) -> str:
    """
    Return the hex color code corresponding to an infringement level label.

    Args:
        infringement_label: One of 'HIGH', 'MEDIUM', or 'LOW'.

    Returns:
        Hex color string ('#RRGGBB').
    """
    if infringement_label == SCORE_LABEL_HIGH:
        return BADGE_HIGH_COLOR
    if infringement_label == SCORE_LABEL_MEDIUM:
        return BADGE_MEDIUM_COLOR
    return BADGE_LOW_COLOR


def render_infringement_badge(infringement_label: str) -> str:
    """
    Produce an HTML span element styled as a colored badge for the given label.

    Args:
        infringement_label: One of 'HIGH', 'MEDIUM', or 'LOW'.

    Returns:
        HTML string containing a styled <span> badge element.
    """
    color = get_badge_color_for_label(infringement_label)
    return (
        f'<span style="background-color:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.8em;font-weight:bold;">'
        f"{infringement_label}</span>"
    )


def render_score_percentage(infringement_score: float) -> str:
    """
    Format a 0.0–1.0 infringement score as a percentage string for display.

    Args:
        infringement_score: Normalized float score in [0.0, 1.0].

    Returns:
        Formatted percentage string, e.g., '83%'.
    """
    return f"{int(infringement_score * 100)}%"


# =============================================================================
# STEP HEADER
# =============================================================================


def render_step_header(step_number: int, step_title: str, completed: bool = False) -> None:
    """
    Display a numbered step header with an optional completion checkmark.

    Args:
        step_number: The step's sequential number in the pipeline.
        step_title: Descriptive title for the pipeline step.
        completed: If True, appends a green checkmark to indicate completion.
    """
    checkmark = " ✅" if completed else ""
    st.subheader(f"Step {step_number}: {step_title}{checkmark}")


# =============================================================================
# PATENT SUMMARY CARD
# =============================================================================


def render_patent_summary_card(
    patent_number: str,
    title: str,
    priority_date: str,
    assignee: str,
    domain: str,
    industry: str,
    selected_claim_numbers: list[int],
) -> None:
    """
    Render a styled summary card for the patent being analyzed.

    Displays key bibliographic fields in a two-column layout using st.metric()
    and st.markdown() within a styled container.

    Args:
        patent_number: The normalized patent identifier.
        title: Patent title string.
        priority_date: Patent priority date string.
        assignee: Patent assignee/owner name.
        domain: Technology domain classification.
        industry: Industry classification.
        selected_claim_numbers: List of selected claim numbers for analysis.
    """
    claims_str = ", ".join(
        f"Claim {n}" for n in selected_claim_numbers) or "None selected"

    st.markdown(
        """
        <div style="background-color:#1e2130;padding:16px;border-radius:10px;
        border-left:4px solid #4a90e2;margin-bottom:16px;">
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Patent Number", patent_number)
        st.metric("Priority Date", priority_date or "Unknown")
    with col2:
        st.metric("Industry", industry)
        st.metric("Domain", domain)
    with col3:
        st.metric("Assignee", assignee or "Unknown")
        st.metric("Analyzing", claims_str)

    if title:
        st.markdown(f"**Title:** {title}")

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# CPC CODES TABLE
# =============================================================================


def render_cpc_codes_table(cpc_codes: list[dict[str, str]]) -> None:
    """
    Display CPC classification codes in a compact Streamlit table.

    Args:
        cpc_codes: List of dicts with 'code' and 'label' keys.
    """
    if not cpc_codes:
        st.info("No CPC codes found for this patent.")
        return
    df = pd.DataFrame(cpc_codes)
    df.columns = ["Code", "Description"]
    st.dataframe(df, use_container_width=True, hide_index=True)


# =============================================================================
# CLAIM DISPLAY
# =============================================================================


def render_independent_claims_checkboxes(
    independent_claims: list[dict],
    session_key_prefix: str = "claim_sel",
) -> list[int]:
    """
    Render checkboxes for selecting which independent claims to analyze.

    Pre-checks all claims by default. Uses Streamlit session state to persist
    selections across reruns.

    Args:
        independent_claims: List of claim dicts with 'claim_number', 'text', 'breadth'.
        session_key_prefix: Prefix for session state keys to avoid collisions.

    Returns:
        List of selected claim numbers.
    """
    st.markdown(
        "**Select which independent claim(s) to analyze for infringement detection.**  "
        "The selected claim(s) will be used for the final infringement decision."
    )
    selected_claim_numbers: list[int] = []

    for claim in independent_claims:
        claim_number = claim.get("claim_number", 0)
        breadth = claim.get("breadth", "")
        claim_text = claim.get("text", "")
        preview = claim_text[:120] + \
            "..." if len(claim_text) > 120 else claim_text

        state_key = f"{session_key_prefix}_{claim_number}"
        if state_key not in st.session_state:
            st.session_state[state_key] = True

        is_selected = st.checkbox(
            f"Claim {claim_number} ({breadth}) — {preview}",
            key=state_key,
        )
        if is_selected:
            selected_claim_numbers.append(claim_number)

    return selected_claim_numbers


def render_dependent_claims_table(dependent_claims: list[dict]) -> None:
    """
    Display dependent claims in a compact expandable table.

    Args:
        dependent_claims: List of claim dicts with 'claim_number', 'text', 'references_claim'.
    """
    if not dependent_claims:
        st.info("No dependent claims found.")
        return
    rows = []
    for claim in dependent_claims:
        rows.append({
            "Claim #": claim.get("claim_number", ""),
            "References Claim": claim.get("references_claim", ""),
            "Text (Preview)": (claim.get("text", "")[:200] + "..."),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# =============================================================================
# CLAIM ELEMENT CHECKLIST
# =============================================================================


def render_claim_elements_checklist(
    elements: list[dict[str, str]],
    session_key_prefix: str = "elem_sel",
) -> list[dict[str, str]]:
    """
    Render an interactive checklist of decomposed claim elements.

    All elements are pre-checked by default. Returns only the elements
    whose checkboxes remain checked.

    Args:
        elements: List of dicts with 'component' and 'function' keys.
        session_key_prefix: Prefix for session state keys.

    Returns:
        List of selected (checked) element dicts.
    """
    st.markdown(
        "**Confirm the claim elements to be used for infringement analysis:**")
    selected_elements: list[dict[str, str]] = []

    for idx, elem in enumerate(elements):
        component = elem.get("component", "")
        function = elem.get("function", "")
        state_key = f"{session_key_prefix}_{idx}"
        if state_key not in st.session_state:
            st.session_state[state_key] = True

        is_selected = st.checkbox(
            f"{component} — {function}",
            key=state_key,
        )
        if is_selected:
            selected_elements.append(elem)

    return selected_elements


# =============================================================================
# GENERATED QUERIES DISPLAY
# =============================================================================


def render_generated_queries(
    company_queries: list[str],
    product_queries: list[str],
) -> None:
    """
    Display the generated SerpAPI search queries in two collapsible sections.

    Args:
        company_queries: List of Set A (company discovery) query strings.
        product_queries: List of Set B (product discovery) query strings.
    """
    col1, col2 = st.columns(2)
    with col1:
        with st.expander(f"🏢 Set A — Company Discovery ({len(company_queries)} queries)", expanded=False):
            for idx, query in enumerate(company_queries, start=1):
                st.markdown(f"{idx}. `{query}`")
    with col2:
        with st.expander(f"📦 Set B — Product Discovery ({len(product_queries)} queries)", expanded=False):
            for idx, query in enumerate(product_queries, start=1):
                st.markdown(f"{idx}. `{query}`")


# =============================================================================
# RESULTS TABLE
# =============================================================================


def render_results_summary_table(analysis_results: list) -> None:
    """
    Display a summary table of all qualifying infringement results.

    Shows Company, Product, Match Level (badge), Score %, and Launch Date
    for each result, sorted by infringement score descending.

    Args:
        analysis_results: List of ProductAnalysisResult objects.
    """
    if not analysis_results:
        st.info("No qualifying product matches found.")
        return

    rows = []
    for result in analysis_results:
        rows.append({
            "Company": result.company_name or "Unknown",
            "Product": result.product_name or "Unknown",
            "Match Level": result.infringement_label,
            "Score": render_score_percentage(result.infringement_score),
            "Score (raw)": result.infringement_score,
            "Matched / Total": f"{result.matched_element_count}/{result.total_elements}",
            "Launch Date": result.product_launch_date or "Unknown",
        })

    df = pd.DataFrame(rows)
    styled = (
        df.drop(columns=["Score (raw)"])
        .style.apply(
            lambda col: [
                (
                    "background-color:#7f1d1d;color:white"
                    if v == SCORE_LABEL_HIGH
                    else "background-color:#78350f;color:white"
                    if v == SCORE_LABEL_MEDIUM
                    else "background-color:#14532d;color:white"
                )
                for v in col
            ],
            subset=["Match Level"],
        )
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# =============================================================================
# INDIVIDUAL RESULT EXPANDER
# =============================================================================


def render_single_result_expander(result, result_index: int) -> None:
    """
    Render an expandable detail panel for a single product infringement result.

    Shows matched/unmatched elements with visual indicators, plain-English
    analysis, product URL, launch date, and company details within an expander.

    Args:
        result: A ProductAnalysisResult dataclass instance.
        result_index: 1-based index for display purposes.
    """
    badge_html = render_infringement_badge(result.infringement_label)
    score_pct = render_score_percentage(result.infringement_score)
    expander_label = (
        f"#{result_index} — {result.company_name or 'Unknown Company'} | "
        f"{result.product_name or 'Unknown Product'} | "
        f"{score_pct} ({result.matched_element_count}/{result.total_elements} elements)"
    )

    with st.expander(expander_label, expanded=False):
        # Header row
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Company:** {result.company_name or 'Unknown'}")
            st.markdown(f"**Product:** {result.product_name or 'Unknown'}")
        with col2:
            st.markdown(
                f"**Infringement Level:** {badge_html}", unsafe_allow_html=True)
            st.markdown(f"**Score:** {score_pct}")
            st.progress(result.infringement_score)
        with col3:
            st.markdown(
                f"**Elements Matched:** {result.matched_element_count}/{result.total_elements}"
            )
            launch_flag = ""
            if result.launch_after_priority_date is True:
                launch_flag = " ⚠️ After priority date"
            elif result.launch_after_priority_date is False:
                launch_flag = " ✅ Before priority date"
            st.markdown(
                f"**Launch Date:** {result.product_launch_date or 'Unknown'}{launch_flag}")

        st.divider()

        # Matched elements
        if result.matched_elements:
            st.markdown("**✅ Matched Claim Elements:**")
            for elem in result.matched_elements:
                st.markdown(f"  - ✅ {elem}")

        # Unmatched elements
        if result.unmatched_elements:
            st.markdown("**❌ Unmatched Claim Elements:**")
            for elem in result.unmatched_elements:
                st.markdown(f"  - ❌ {elem}")

        st.divider()

        # Plain English analysis
        if result.plain_english_analysis:
            st.markdown("**Plain-English Analysis:**")
            st.info(result.plain_english_analysis)

        # Source URL
        st.markdown(f"**Source URL:** [{result.url}]({result.url})")


# =============================================================================
# FULL RESULTS DASHBOARD
# =============================================================================


def render_full_results_dashboard(
    analysis_results: list,
    patent_number: str,
    title: str,
    priority_date: str,
    domain: str,
    industry: str,
    selected_claim_numbers: list[int],
    assignee: str,
) -> None:
    """
    Render the complete results dashboard including summary card, metrics,
    results table, and individual expanded detail panels.

    Args:
        analysis_results: List of qualifying ProductAnalysisResult objects.
        patent_number: The patent identifier.
        title: Patent title.
        priority_date: Patent priority date.
        domain: Technology domain.
        industry: Industry classification.
        selected_claim_numbers: List of claim numbers analyzed.
        assignee: Patent assignee name.
    """
    st.markdown("---")
    render_step_header(10, "Infringement Analysis Results", completed=True)

    render_patent_summary_card(
        patent_number=patent_number,
        title=title,
        priority_date=priority_date,
        assignee=assignee,
        domain=domain,
        industry=industry,
        selected_claim_numbers=selected_claim_numbers,
    )

    if not analysis_results:
        st.warning(
            "No qualifying product matches were identified. Try adjusting claim element selection or re-running with different queries.")
        return

    # Top-level metrics
    high_count = sum(
        1 for r in analysis_results if r.infringement_label == SCORE_LABEL_HIGH)
    medium_count = sum(
        1 for r in analysis_results if r.infringement_label == SCORE_LABEL_MEDIUM)
    total_count = len(analysis_results)

    low_count = total_count - high_count - medium_count
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Matches Found", total_count)
    m2.metric("🔴 HIGH Risk", high_count)
    m3.metric("🟠 MEDIUM Risk", medium_count)
    m4.metric("🟡 LOW Risk", low_count)

    st.markdown("### Results Summary")
    render_results_summary_table(analysis_results)

    st.markdown("### Detailed Analysis")
    for idx, result in enumerate(analysis_results, start=1):
        render_single_result_expander(result, idx)


# =============================================================================
# CSV EXPORT
# =============================================================================


def build_results_dataframe_for_export(analysis_results: list) -> pd.DataFrame:
    """
    Convert a list of ProductAnalysisResult objects into a flat DataFrame for CSV export.

    Args:
        analysis_results: List of ProductAnalysisResult instances.

    Returns:
        pandas DataFrame with one row per result, suitable for st.download_button.
    """
    rows = []
    for result in analysis_results:
        rows.append({
            "URL": result.url,
            "Company": result.company_name,
            "Product": result.product_name,
            "Infringement Level": result.infringement_label,
            "Infringement Score": f"{result.infringement_score:.2f}",
            "Score (%)": render_score_percentage(result.infringement_score),
            "Matched Elements Count": result.matched_element_count,
            "Total Elements": result.total_elements,
            "Matched Elements": "; ".join(result.matched_elements),
            "Unmatched Elements": "; ".join(result.unmatched_elements),
            "Product Launch Date": result.product_launch_date,
            "Launch After Priority Date": result.launch_after_priority_date,
            "Plain English Analysis": result.plain_english_analysis,
        })
    return pd.DataFrame(rows)


def render_csv_export_button(analysis_results: list, patent_number: str) -> None:
    """
    Render a Streamlit download button for exporting results as a CSV file.

    Args:
        analysis_results: List of ProductAnalysisResult instances.
        patent_number: Used to construct the filename.
    """
    if not analysis_results:
        return
    df = build_results_dataframe_for_export(analysis_results)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    filename = f"ip8_results_{patent_number}.csv"
    st.download_button(
        label="📥 Download Full Results as CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


# =============================================================================
# PROGRESS HELPERS
# =============================================================================


def render_pipeline_progress(step_label: str, progress_fraction: float) -> None:
    """
    Display a labeled progress bar for a named pipeline step.

    Args:
        step_label: Descriptive label shown above the progress bar.
        progress_fraction: Float in [0.0, 1.0] representing completion.
    """
    st.markdown(f"**{step_label}**")
    st.progress(min(1.0, max(0.0, progress_fraction)))


def render_cache_info_banner(patent_number: str, cache_size_bytes: int) -> None:
    """
    Display an informational banner indicating results were loaded from cache.

    Args:
        patent_number: The patent identifier loaded from cache.
        cache_size_bytes: Total cache file size in bytes.
    """
    size_kb = cache_size_bytes / 1024
    st.info(
        f"✅ Loaded from cache for **{patent_number}** "
        f"({size_kb:.1f} KB cached). "
        "Use the sidebar to clear cache and re-run."
    )
