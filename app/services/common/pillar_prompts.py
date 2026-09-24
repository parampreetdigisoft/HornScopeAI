"""
Data Analyzer Service - LLM-powered analysis of SQL Server data
Enhanced with HornScope Platform (HSP) pillar prompts.
Pillars are loaded dynamically from the database — not hardcoded.
"""

from typing import Dict, List, Mapping, Optional, Union

_PILLAR_FEED_JSON_RULES = """
        Return ONLY valid JSON.
        - Output must start with { and end with }
        - No markdown, code fences, or text outside JSON
        - Use double quotes only; no trailing commas
        """

_PILLAR_FEED_OUTPUT_STYLE = """
        - Write for a general audience (no technical jargon)
        - Use clear, concise statements; no bullet lists inside JSON strings
        """

PillarRecord = Dict[str, Union[int, str, None]]


class HSPillarPrompts:
    """Provides HSP governance rules and dynamic pillar context from database records."""

    GOVERNANCE_PROTOCOL = """
        =============================================================================
        AI MASTER GOVERNANCE PROTOCOL (HSP) — MANDATORY FOR EVERY ASSESSMENT
        HornScope Platform
        =============================================================================

        1. DATA INPUTS FOR COUNTRY TRAJECTORY AND STRATEGIC INTELLIGENCE
        HSP ingests and correlates multi-source data for the Horn of Africa and East Africa.
        These data streams are updated on rolling cycles and standardized prior to modeling.
        Approved source categories:
        - Official Documents: Government reports, parliamentary records, policy documents,
          budget statements.
        - Security Reports: Conflict monitoring, terrorism tracking, peace agreement monitoring.
        - Economic Data: IMF, World Bank, central bank reports; market data; trade statistics.
        - Media & Real-Time Reporting: Major regional and international media, specialist
          intelligence publications (context and recency only; never primary evidence).
        - Diplomatic Records: Foreign ministry statements, diplomatic cables, treaty registrations.
        - Civil Society Reports: NGO assessments, human rights reporting, local governance
          monitoring.
        - Satellite & Geospatial: Infrastructure assessment, agricultural monitoring,
          disaster tracking.
        - Academic & Research: Peer-reviewed research, think tank analysis, policy papers.

        2. AI DISCOVERY PROTOCOL
        For every assessment, HSP must:
        1. Identify the assessment context (country, time period, focus areas).
        2. Generate search queries based on the context and pillar structure.
        3. Execute autonomous web searches using approved sources.
        4. Retrieve and download documents in original format.
        5. Parse and process documents using layout-aware parsing and extraction.
        6. Store processed documents in a vector database for evidence retrieval.
        7. Generate a "Discovery Report" listing all sources found with metadata.
        8. Flag missing or unavailable sources for human follow-up (Stage 2).

        3. DISCOVERY QUALITY ASSURANCE
        Apply these targets before using sources in an assessment. If a metric is
        below target, take the stated action:
        - Source Coverage: ≥90% of known relevant sources (Action if below: AI re-runs search with alternative queries).
        - Source Freshness: Sources from within last 12 months (Action if below: Flag older sources for human review).
        - Source Authority: ≥80% from high-authority sources (Action if below: Flag low-authority sources for human verification).
        - Document Retrieval: ≥95% retrieval rate (Action if below: Log failed retrievals for human follow-up).
        - Multi-Source Balance: No single category dominates (>50%) (Action if below: Suggest additional search directions).

        4. AI MODELING ARCHITECTURE
        HSP employs an ensemble modeling approach, combining:
        - Tree-based machine-learning models for non-linear pattern detection across
          geopolitical, security, governance, and economic signals.
        - Neural networks for interaction effects among HornScope domains.
        - Time-series forecasting for conflict intensity, fiscal stress, prices, and displacement.
        - Anomaly-detection algorithms for abrupt security, political, corridor, or cyber shocks.
        - Network models of elite capture, regional alignment, and corridor exposure.
        Individual model outputs are combined into a composite country-trajectory
        assessment through ensemble weighting. Models are retrained on rolling windows,
        back-tested against historical strategic shocks, and monitored for performance drift.

        5. PREDICTION HORIZONS
        - Near-term operational: 7-30 days
        - Short-term: 1-4 weeks
        - Medium-term: 1-3 months
        - Strategic foresight: 12-24 months

        6. PREDICTIVE OUTPUTS (COUNTRY TRAJECTORY)
        Assess directional risk across these ten HornScope predictions:
        1. Geopolitical and regional-order risk - neighbour relations, external interference,
           and loss of strategic autonomy.
        2. Armed conflict and security escalation - organised violence, terrorism, unrest,
           and erosion of the monopoly on legitimate force.
        3. Governance and rule-of-law erosion - political instability, contested elections,
           and weakening courts.
        4. Macro-fiscal stress - revenue shortfalls, debt-service pressure, and public-finance
           fragility.
        5. Trade, infrastructure, and corridor disruption - ports, borders, highways, and
           connectivity failures.
        6. Social cohesion and human-development strain - demographic pressure, exclusion,
           and service-delivery breakdown.
        7. Climate and natural-resource shock - drought, flood, resource contestation, and
           environmental stress.
        8. Cyber, technology, and information-space breakdown - outages, surveillance,
           disinformation, and narrative capture.
        9. Humanitarian resilience failure - displacement, protection gaps, and weak shock
           absorption.
        10. Country Trajectory Classification - Strong; Functional; Strained; Weak; or Critical.
        Each assessment must name dominant contributing signals and what they mean for
        governments, intelligence, investors, and development partners in the Horn of Africa
        and East Africa.

        7. INTEGRATION WITH HORNSCOPE PILLARS
        Trajectory risk is cross-referenced with HSP pillar capacity across geopolitics,
        security, governance, macroeconomy, trade and connectivity, society, climate,
        technology, narratives, humanitarian resilience, and strategic foresight. This
        converts prediction into actionable strategic intelligence.

        8. EVIDENCE HIERARCHY (priority order)
        L1: National laws, budgets, audits, treaties, official security and diplomatic notices
        L2: National authorities, auditor-general, electoral bodies, regulators
        L3: IMF, World Bank, AfDB, AU, IGAD, UN, and official IFI/REC staff reports
        L4: Peer-reviewed research and validated strategic or investment-climate assessments
        L5: Civil society, human-rights monitors, chambers, and local-governance reporting
        L6: Technical, satellite, logistics, and cyber-incident data
        L7: Media (context only, never primary)
        Rules:
        - ≥2 independent sources per claim
        - No single-source conclusions
        - Structural/operational evidence > perception

        9. FOUR-LAYER EVIDENCE (ALL REQUIRED)
        a) Structural (laws, institutions, treaties, mandates, constitutional order)
        b) Operational (budgets, enforcement, security operations, administration, customs)
        c) Outcome (conflict incidents, displacement, growth, corridor throughput, measured results)
        d) Perception (public trust, grievance, elite and social chatter)
        → Perception cannot override structural/operational evidence

        10. DISTRIBUTIONAL ANALYSIS (MANDATORY)
        Test for core vs hinterland gaps, corridor vs interior access, connected vs excluded
        communities, and urban vs rural operability. Severe capture or exclusion
        is a material structural weakness.

        11. DATA SILENCE RULE
        - Assign Indeterminate/N/A only for the unverifiable indicator, not the whole country
        - Do NOT set ai_score or ai_progress to 0 because some sources are lagged or missing.
          Unknown does not reduce a numeric score that other evidence already supports.
        - 0 means verified absence or a destabilizing condition, not missing data.
        - State cause (conflict, suppression, incapacity, missing systems)
        - Treat silence as governance risk — silence ≠ success
        - Flag gaps in opacity_risk; still return a numeric country score from available evidence

        12. CONTINUOUS LEARNING AND QUALITY ASSURANCE
        - Quarterly back-testing and performance reporting
        - Drift detection triggers retraining
        - Accuracy metrics (AUC, precision, recall, Brier score) tracked over time
        - Prediction audit trail maintained

        13. DESIGN PHILOSOPHY
        HSP prioritizes early sensitivity for high-impact strategic shocks in the Horn of
        Africa and East Africa (conflict escalation, governance rupture, corridor collapse,
        climate compounding, humanitarian breakdown), accepting limited false positives to
        minimize missed decision-critical events. The system favors truthful uncertainty
        over artificial certainty, presenting probabilities and confidence levels rather
        than binary claims.

        14. PROHIBITIONS
        Do NOT:
        - Present deterministic collapse or regime-change predictions without probability
          and confidence
        - Use rankings as analysis
        - Reward opacity or missing official data
        - Accept claims without verification
        - Treat announced reforms as measured outcomes
        - Use media as primary evidence
        - Frame analysis as clinical diagnosis; humanitarian and resilience analysis is in scope

        GLOBAL TERMINOLOGY RULE
        - Always refer to every assessment "Pillar" as a "Domain."
        - Replace all occurrences of "Pillar" with "Domain" unless the term appears in an official title, database field, API response, or direct user quotation.
        - Apply these terminology rules consistently throughout every generated response.
        - Treat every generated report as an annual analytical report, not a news article.
        - Never describe events as if they are unfolding in real time.
        - Avoid uncertain or speculative wording.
        - Use formal, evidence-based, retrospective language.
        - Reference the reporting period instead of "recently" or "over the past weeks."
        - State verified observations and explain their significance.
        - Include specific actors and geographic scope whenever possible.
        - Distinguish confirmed findings from uncertainty. If evidence is insufficient, state that directly instead of using speculative phrases.
        =============================================================================
    """

    @staticmethod
    def _normalize_pillars(
        pillars: Union[Mapping[int, PillarRecord], List[PillarRecord], None],
    ) -> Dict[int, PillarRecord]:
        if not pillars:
            return {}

        if isinstance(pillars, list):
            return {
                int(p["PillarID"]): p
                for p in pillars
                if p.get("PillarID") is not None
            }

        return {int(pid): p for pid, p in pillars.items()}

    @classmethod
    def format_pillar_context(cls, pillar_name: str, description: Optional[str] = None) -> str:
        """Build pillar context from database name and description."""
        desc = (description or "").strip() or "No description provided for this pillar."
        return (
            f"PILLAR: {pillar_name}\n\n"
            f"DESCRIPTION:\n{desc}\n\n"
            f"ASSESSMENT GUIDANCE:\n"
            f"Evaluate this pillar using the description above, the HSP governance protocol, "
            f"and verifiable strategic evidence for the target Horn of Africa or East "
            f"Africa country. Focus on structural capacity, operational delivery, "
            f"measured outcomes, and equity of access across geography and communities."
        )

    @classmethod
    def get_pillar_context(
        cls,
        pillar_id: int,
        pillars: Union[Mapping[int, PillarRecord], List[PillarRecord], None] = None,
        *,
        pillar_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """Return formatted context for a pillar using DB records or explicit name/description."""
        pillar_map = cls._normalize_pillars(pillars)
        pillar = pillar_map.get(pillar_id)
        if pillar:
            return cls.format_pillar_context(
                str(pillar.get("PillarName") or pillar_name or f"Pillar {pillar_id}"),
                pillar.get("Description") or description,
            )

        if pillar_name:
            return cls.format_pillar_context(pillar_name, description)

        return f"No context available for pillar ID {pillar_id}."

    @classmethod
    def get_all_pillar_names(
        cls,
        pillars: Union[Mapping[int, PillarRecord], List[PillarRecord], None] = None,
    ) -> Dict[int, str]:
        """Return a mapping of pillar ID to pillar name from database records."""
        pillar_map = cls._normalize_pillars(pillars)
        return {
            pid: str(p.get("PillarName", f"Pillar {pid}"))
            for pid, p in sorted(pillar_map.items())
        }

    @classmethod
    def get_pillar_catalog_for_live_feed(
        cls,
        pillars: Union[Mapping[int, PillarRecord], List[PillarRecord], None] = None,
    ) -> str:
        """Compact HSP pillar catalog for live pillar signals."""
        pillar_map = cls._normalize_pillars(pillars)
        if not pillar_map:
            return "No active pillars configured."

        lines = []
        for pid in sorted(pillar_map.keys()):
            pillar = pillar_map[pid]
            name = str(pillar.get("PillarName", f"Pillar {pid}"))
            description = str(pillar.get("Description") or "").strip()
            focus = description[:280].strip() if description else name
            lines.append(
                f"Pillar {pid} — {name}\n"
                f"  Focus: {focus}"
            )
        return "\n\n".join(lines)

    @classmethod
    def pillar_live_signals_prompt(
        cls,
        pillars: Union[Mapping[int, PillarRecord], List[PillarRecord], None] = None,
    ) -> str:
        pillar_map = cls._normalize_pillars(pillars)
        pillar_ids = sorted(pillar_map.keys())
        pillar_count = len(pillar_ids)
        id_range = (
            f"{pillar_ids[0]} through {pillar_ids[-1]}"
            if pillar_count > 1
            else str(pillar_ids[0]) if pillar_ids else "none"
        )
        catalog = cls.get_pillar_catalog_for_live_feed(pillar_map)
        example_id = pillar_ids[0] if pillar_ids else 1

        return f"""
        You are the HornScope Platform (HSP) live pillar intelligence engine.

        Produce a LIVE HornScope-focused snapshot: exactly ONE card per active HSP pillar.
        Use the pillar definitions below to ground each card in the correct HornScope domain.

        ==================================================
        HSP PILLAR CATALOG (ALL {pillar_count} — MANDATORY COVERAGE)
        ==================================================
        {catalog}

        ==================================================
        MANDATORY: LIVE WEB SEARCH
        ==================================================
        Before writing JSON, search credible Horn of Africa and East Africa news for each pillar.
        For each pillar, find the most relevant signal from the LAST 48 HOURS affecting
        geopolitics, peace and security, governance, public finance, trade corridors,
        society and human development, climate and natural resources, technology and cyber,
        or regional strategic outlook. Older context only if an actively developing
        conflict, diplomatic, fiscal, or humanitarian story requires brief background.

        Prefer: Reuters, BBC, Al Jazeera, The East African, IGAD, African Union, UN/OCHA,
        ReliefWeb, Crisis Group, IMF/World Bank/AfDB updates, and established regional
        trackers. Do not invent article URLs.

        ==================================================
        sourceUrl RULES
        ==================================================
        - One HTTPS URL per pillar, copied exactly from search OR Google News search:
          https://news.google.com/search?q=PILLAR+TOPIC+KEYWORDS+HORN+OF+AFRICA+OR+EAST+AFRICA&hl=en-US&gl=US&ceid=US:en
        - NEVER fabricate article slugs on Reuters, BBC, Al Jazeera, UN, IGAD, AU, etc.

        ==================================================
        OUTPUT RULES
        ==================================================
        - Return EXACTLY {pillar_count} pillar objects (pillarId {id_range}, each once).
        - title: max 55 characters — headline-style.
        - summary: max 100 characters — one clear strategic signal for this pillar.
        - type: "risk" or "trend" (lowercase).
        - status: Rising | Active | Watch | Stable | Critical
        - urgency: low | medium | high | critical
        - color: green | yellow | orange | red | blue
        - Do NOT mention source names in title or summary.
        - headline/subHeadline: live 48-hour framing for HornScope strategic intelligence.
        - updatedAt: current UTC ISO-8601.


        JSON format:
        {{
            "updatedAt": "2026-05-25T12:00:00Z",
            "headline": "Live Pillar Signals",
            "subHeadline": "HornScope pillar watch from the last 48 hours.",
            "pillars": [
                {{
                    "pillarId": {example_id},
                    "type": "risk",
                    "title": "Short headline",
                    "summary": "One sentence strategic signal for this pillar domain.",
                    "status": "Watch",
                    "urgency": "medium",
                    "color": "yellow",
                    "sourceUrl": "https://news.google.com/search?q=Horn+of+Africa+OR+East+Africa+security&hl=en-US&gl=US&ceid=US:en"
                }}
            ]
        }}

        {_PILLAR_FEED_OUTPUT_STYLE}
        {_PILLAR_FEED_JSON_RULES}
        """

    @classmethod
    def pillar_overview_prompt(
        cls,
        pillars: Union[Mapping[int, PillarRecord], List[PillarRecord], None] = None,
    ) -> str:
        """One summary paragraph and improvement points for each active pillar."""
        pillar_map = cls._normalize_pillars(pillars)
        pillar_count = len(pillar_map)
        catalog = cls.get_pillar_catalog_for_live_feed(pillar_map)
        example_id = min(pillar_map.keys()) if pillar_map else 1

        return f"""
        You are the HornScope public domain overview engine.
        Return exactly one entry for each of the {pillar_count} pillars below.
        summary: one paragraph. Name the leading Horn of Africa or East Africa countries in that paragraph.
        areasForImprovement: two or three sentences on the main areas for improvement.
        Do not add a leaders list or any field other than pillarId, pillarName, summary, and areasForImprovement.
        Do not invent countries outside that region. No bullet lists inside JSON strings.

        PILLARS:
        {catalog}

        JSON format:
        {{
            "pillars": [
                {{
                    "pillarId": {example_id},
                    "pillarName": "Name of the pillar",
                    "summary": "One paragraph on this domain.",
                    "areasForImprovement": "Two or three sentences on the main areas for improvement."
                }}
            ]
        }}

        {_PILLAR_FEED_OUTPUT_STYLE}
        {_PILLAR_FEED_JSON_RULES}
        """

    @staticmethod
    def pillar_overview_user_prompt() -> str:
        return """
        Current UTC datetime (now):
        {current_date}

        Write the domain overview JSON now.
        """.strip()
