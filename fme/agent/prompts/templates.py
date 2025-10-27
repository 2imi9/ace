"""Prompt templates for dataset and simulation briefings."""

from __future__ import annotations

from textwrap import dedent

# General template for briefing the language model.
BASE_BRIEFING_TEMPLATE = dedent(
    """
    You are an analytical assistant helping with {analysis_type_display}.

    ### Dataset Overview
    - **Name:** {dataset_name}
    - **Description:** {dataset_overview}

    ### Simulation Metadata
    - **Objective:** {simulation_objective}
    {simulation_metadata_section}

    ### Variable Catalog
    {variable_section}

    ### Simulation Assumptions
    {assumption_bullets}

    ### Uncertainty and Limitations
    {uncertainty_section}

    ### Analysis Directives
    {analysis_focus}

    ### Requested Outputs
    {requested_outputs}

    ### User Request
    {user_request}

    ### Additional Context
    {additional_context}
    """
)

# Specialized analysis directives for different workflows.
SPECIALIZED_ANALYSIS_TEMPLATES = {
    "exploratory_data_analysis": dedent(
        """
        Perform exploratory data analysis. Prioritize:
        - Overview of distributions, ranges, and central tendencies.
        - Identification of missing values, outliers, or data quality issues.
        - Relationships between key variables using correlations or cross-tabulations.
        - Visual or descriptive summaries that highlight noteworthy structure in the data.
        """
    ),
    "hypothesis_generation": dedent(
        """
        Generate plausible, testable hypotheses grounded in the dataset and simulation context. Include:
        - Potential causal mechanisms and supporting evidence from the metadata.
        - Variables likely involved in each hypothesis and the rationale behind their roles.
        - Suggestions for analyses or experiments that could validate or refute each hypothesis.
        """
    ),
    "result_interpretation": dedent(
        """
        Interpret existing simulation or analysis results. Focus on:
        - Connecting the reported outcomes back to assumptions and scenario settings.
        - Assessing how uncertainty sources may influence the conclusions.
        - Highlighting implications, limitations, and recommended follow-up actions.
        """
    ),
}

DEFAULT_ANALYSIS_TEMPLATE = dedent(
    """
    Provide a concise, technically rigorous response that references the supplied metadata.
    """
)
