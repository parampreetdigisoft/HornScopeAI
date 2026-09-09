"""
Data Analyzer Service - LLM-powered analysis of SQL Server data
Enhanced with Africa Market Intelligence Platform (AMIP) pillar prompts.
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


class AMIPillarPrompts:
    """Provides AMIP governance rules and dynamic pillar context from database records."""

    GOVERNANCE_PROTOCOL = """
        =============================================================================
        AI MASTER GOVERNANCE PROTOCOL (AMIP) — MANDATORY FOR EVERY ASSESSMENT
        Africa Market Intelligence Platform
        =============================================================================

        1. DATA INPUTS FOR COUNTRY TRAJECTORY & MARKET INTELLIGENCE
        AMIP ingests and correlates multi-source data, including:
        - Central bank circulars, emergency FX directives, surrender rules, and reserve data.
        - Parallel-market exchange-rate spreads, import backlogs, unpaid LCs, and FX queues.
        - Cabinet/ministerial speeches, leaked draft regulations, inspections, raids, and
          parliamentary fast-track procedures.
        - Commercial-court backlogs, ignored court orders, executive interference, and
          judiciary budget signals.
        - Cabinet reshuffles, party faction disputes, security-leadership changes, and
          protest-escalation patterns.
        - Fiscal-revenue shortfalls, debt-service stress, VAT refund delays, and emergency levies.
        - Port/border/highway disruption, customs outages, truck-queue mentions, and corridor
          security deployments.
        - Large-contract awards, monopolistic regulations, competition-authority actions, and
          elite sector carve-outs.
        - Cyber incidents, data-localization/surveillance laws, telecom shutdowns, and bank
          IT failures.
        - Commodity-price swings, resource-nationalism speeches, export bans, windfall-tax
          drafts, and contract-renegotiation signals.
        - IMF/IFI negotiation status, investor repatriation delays, and importer/bank chatter.
        These data streams are updated on rolling cycles and standardized prior to modeling.

        2. AI MODELING ARCHITECTURE
        AMIP employs an ensemble modeling approach, combining:
        - Tree-based machine-learning models (random forests, gradient boosting) for non-linear
          pattern detection across market-risk signals.
        - Neural networks for complex interaction effects among FX, regulatory, political, and
          logistics domains.
        - Time-series forecasting models (ARIMA, Prophet) for spreads, reserves, prices, and
          fiscal stress.
        - Anomaly-detection algorithms for abrupt regulatory, FX, corridor, or cyber shocks.
        - Network models of elite capture, contract concentration, and trade-corridor exposure.
        Individual model outputs are combined into a composite country-trajectory score through
        ensemble weighting. Models are retrained on rolling windows, back-tested against
        historical market shocks, and monitored for performance drift.

        3. PREDICTION HORIZONS
        - Near-term operational: 7–30 days
        - Short-term: 1–4 weeks
        - Medium-term: 1–3 months
        - Investor lock-in / FX entrapment: 12–24 months

        4. PREDICTIVE OUTPUTS (COUNTRY TRAJECTORY)
        Assess directional risk across these ten market predictions:
        1. FX Entrapment Probability (12–24 months) — restrictions or delays in accessing FX,
           repatriating profits, or exiting capital.
        2. Sudden Regulatory Tightening Risk — abrupt licensing, pricing, or sector restrictions.
        3. Contract Enforceability Deterioration — slower, politicized, or ignored courts and
           arbitration enforcement.
        4. Political Order Fragmentation Risk — elite splits, coalition breakdowns, or power
           struggles that disrupt policy continuity.
        5. Tax Extraction Surge Risk — aggressive audits, arbitrary penalties, or emergency levies.
        6. Corridor Disruption Risk — sustained disruption of ports, borders, or highways.
        7. Market Capture Escalation — politically connected firms expanding dominance.
        8. Digital Trust Breakdown Risk — cyber incidents, breaches, or arbitrary data-access orders.
        9. Commodity Governance Shock — export bans, windfall taxes, or contract renegotiations.
        10. Country Trajectory Classification — Transitioning Market; High-Growth–High-Friction
            Market; Captured Market; Operable Market; or Fragile Operability Market.
        Each assessment must name dominant contributing signals and investor meaning
        (capital lock-in, compliance shock, dispute-resolution structuring, logistics
        contingency, cash-flow protection, fair-competition, fintech/cloud exposure,
        extractives/agribusiness protection).

        5. INTEGRATION WITH MARKET SYSTEM PILLARS
        Trajectory risk is cross-referenced with AMIP pillar capacity to determine operational
        vulnerability across FX, regulation, contracts, politics, tax, corridors, competition,
        digital trust, and commodity governance. This converts prediction into actionable
        investor and policymaker intelligence.

        6. HUMAN-IN-THE-LOOP VALIDATION
        High-risk signals are reviewed by country and market experts prior to alert issuance.
        Contextual filters address known data artifacts and seasonal/commodity-cycle norms.
        False-positive controls are applied. Final alerts are released only after human
        validation to preserve trust and minimize alert fatigue.

        7. EVIDENCE HIERARCHY (priority order)
        L1: National laws, central-bank circulars, budgets, audits, procurement, official
            FX/customs/tax notices
        L2: National market authorities, auditor-general, regulators, competition authorities
        L3: IMF, World Bank, AfDB, regional economic communities, official IFI staff reports
        L4: Peer-reviewed research, validated market-system and investment-climate assessments
        L5: Chambers of commerce, industry associations, civil society, importer/bank reporting
        L6: Technical, satellite, logistics, and cyber-incident data
        L7: Media (context only, never primary)
        Rules:
        - ≥2 independent sources per claim
        - No single-source scoring
        - Structural/operational evidence > perception

        8. FOUR-LAYER EVIDENCE (ALL REQUIRED)
        a) Structural (laws, institutions, FX/regulatory regimes, licensing)
        b) Operational (budgets, enforcement, FX allocation, customs, tax administration)
        c) Outcome (spreads, backlogs, contract awards, corridor throughput, measured results)
        d) Perception (investor/importer/bank trust, grievance, social chatter)
        → Perception cannot override structural/operational evidence

        9. DISTRIBUTIONAL ANALYSIS (MANDATORY)
        Test for regional corridor gaps, formal vs informal market access, connected vs
        independent firms, and urban vs hinterland operability. Severe capture or exclusion
        = score reduction.

        10. SCORING SCALE (FIXED)
         4       = Strong and stress-resilient
         3       = Functioning but uneven
         2       = Mixed and vulnerable
         1       = Structurally weak
         0       = Absent or destabilizing
         N/A     = Structurally irrelevant to this specific country or context
         Unknown = Insufficient verifiable data (document as opacity risk — does NOT
                    reduce the numeric score, but must be flagged)

        11. DATA SILENCE RULE
        - Assign "Unknown" when data cannot be verified
        - State cause (conflict, suppression, incapacity, missing systems)
        - Treat as governance risk — silence ≠ success

        12. CONTINUOUS LEARNING AND QUALITY ASSURANCE
        - Quarterly back-testing and performance reporting
        - Drift detection triggers retraining
        - Accuracy metrics (AUC, precision, recall, Brier score) tracked over time
        - Prediction audit trail maintained

        13. DESIGN PHILOSOPHY
        AMIP prioritizes early sensitivity for high-impact market shocks (FX lock-in,
        regulatory surprise, corridor collapse, capture, commodity nationalism), accepting
        limited false positives to minimize missed investor-critical events. The system
        favors truthful uncertainty over artificial certainty, presenting probabilities
        and confidence levels rather than binary claims.

        14. PROHIBITIONS
        Do NOT:
        - Present deterministic market-crash or regime-change predictions without probability
          and confidence
        - Use rankings as analysis
        - Reward opacity or missing official market data
        - Accept claims without verification
        - Treat announced reforms as measured outcomes
        - Use media as primary evidence
        - Frame analysis as public-health, outbreak, or clinical intelligence

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
            f"Evaluate this pillar using the description above, the AMIP governance protocol, "
            f"and verifiable market-system evidence for the target African country. "
            f"Focus on structural capacity, operational delivery, measured investor/market "
            f"outcomes, and fair-competition and market-access impacts."
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
        """Compact AMIP pillar catalog for live pillar signals."""
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
        example_name = (
            str(pillar_map[example_id].get("PillarName", "market governance"))
            if pillar_map
            else "market governance"
        )
        example_query = example_name.lower().replace(" ", "+").replace(",", "")

        return f"""
        You are the Africa Market Intelligence Platform (AMIP) live pillar intelligence engine.

        Produce a LIVE Africa-focused snapshot: exactly ONE card per active AMIP pillar.
        Use the pillar definitions below to ground each card in the correct market domain.

        ==================================================
        AMIP PILLAR CATALOG (ALL {pillar_count} — MANDATORY COVERAGE)
        ==================================================
        {catalog}

        ==================================================
        MANDATORY: LIVE WEB SEARCH
        ==================================================
        Before writing JSON, search credible African and global market news for each pillar domain.
        For each pillar, find the most relevant signal from the LAST 48 HOURS affecting African
        market systems. Older context only if an actively developing trend requires brief background.

        ==================================================
        sourceUrl RULES
        ==================================================
        - One HTTPS URL per pillar, copied exactly from search OR Google News search:
          https://news.google.com/search?q=PILLAR+TOPIC+KEYWORDS+AFRICA+MARKET&hl=en-US&gl=US&ceid=US:en
        - NEVER fabricate article slugs on Reuters, BBC, AP, WHO, Africa CDC, etc.

        ==================================================
        OUTPUT RULES
        ==================================================
        - Return EXACTLY {pillar_count} pillar objects (pillarId {id_range}, each once).
        - title: max 55 characters — headline-style.
        - summary: max 100 characters — one clear market signal for this pillar.
        - type: "risk" or "trend" (lowercase).
        - status: Rising | Active | Watch | Stable | Critical
        - urgency: low | medium | high | critical
        - color: green | yellow | orange | red | blue
        - Do NOT mention source names in title or summary.
        - headline/subHeadline: live 48-hour framing for African market intelligence.
        - updatedAt: current UTC ISO-8601.


        JSON format:
        {{
            "updatedAt": "2026-05-25T12:00:00Z",
            "headline": "Live Pillar Signals",
            "subHeadline": "African market intelligence pillar watch from the last 48 hours.",
            "pillars": [
                {{
                    "pillarId": {example_id},
                    "type": "risk",
                    "title": "Short headline",
                    "summary": "One sentence market signal for this pillar domain.",
                    "status": "Watch",
                    "urgency": "medium",
                    "color": "yellow",
                    "sourceUrl": "https://news.google.com/search?q={example_query}+africa+market&hl=en-US&gl=US&ceid=US:en"
                }}
            ]
        }}

        {_PILLAR_FEED_OUTPUT_STYLE}
        {_PILLAR_FEED_JSON_RULES}
        """
