# =========================================================================== #
#  rag_query_service.py  (refactored)                                         #
# =========================================================================== #
"""
RAGQueryService  (refactored)
------------------------------
Two-stage RAG pipeline for country document Q&A.

Stage 1 — LLM-driven TOC routing  (which sections are relevant?)
Stage 2 — ChromaDB vector search within those sections

LLM calls are handled by LLMBaseService.
All prompt text comes from HSPromptTemplates.
"""

from datetime import datetime, timedelta, timezone

import os
import re
import chromadb
import logging
import json
import httpx
from typing import List, Dict, Any, Optional
from app.services.common.embedding import create_embedding_function
from app.services.common.llm_base_service import LLMBaseService
from app.services.common.country_prompt import HSPromptTemplates
from app.services.common.freenews_client import (
    FreeNewsRequestError,
    fetch_first_article,
    is_invalid_country,
    note_invalid_country,
)
from app.services.common.pillar_prompts import HSPillarPrompts
from app.services.core.repository import DatabaseRepository
from app.services.common import json_response_parser as jrp
from app.services.common.citation_binder import (
    apply_verified_citations,
    merge_verified_sources
)
from app.services.common.web_search_client import (
    invoke_with_web_search,
    web_search_available,
)
from app.services.common.web_search_policy import question_needs_web_search
logger = logging.getLogger(__name__)

CHROMA_PATH = "./chroma_store"


class RAGQueryService:
    """
    Hybrid RAG service: LLM-routed TOC selection + ChromaDB vector retrieval.

    LLM mechanics live in LLMBaseService (injected).
    Prompt text lives in HSPromptTemplates.
    """

    def __init__(self) -> None:

        # Ensure directory exists
        if not os.path.exists(CHROMA_PATH):
            os.makedirs(CHROMA_PATH)

        try:
            self.client = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=chromadb.config.Settings(anonymized_telemetry=False),
            )

        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            raise

        self.embed_fn = create_embedding_function()
        self._db = DatabaseRepository()
        # --- LLM (shared base service) ---
        self._llm_svc = LLMBaseService(max_retries=3, retry_delay=1.0)

    # ------------------------------------------------------------------ #
    #  Initialisation                                                    #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Initialise the shared LLM service."""
        await self._llm_svc.initialize()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def get_country_document_context(
        self,
        country_id: int,
        msg_text: str,
        pillar_id: Optional[int] = None,
    ) -> str:
        """
        Answer a natural-language question about a country using:
          1. LLM-selected TOC sections
          2. ChromaDB vector search within those sections
          3. LLM synthesis of retrieved chunks + chat history
        """
        # Stage 1 — TOC routing
        toc = await self._get_country_toc(country_id, pillar_id)

        relevant_toc_ids = []
        if len(toc) > 4:
            relevant_toc_ids = await self._get_relevant_Id(msg_text, toc)
        else:
            relevant_toc_ids = [row["TOCID"] for row in toc]

        # Stage 2 — Vector retrieval
        chunks = self._fetch_relevant_chunks(
            question=msg_text,
            toc_ids=relevant_toc_ids,
            country_id=country_id,
            pillar_id=pillar_id,
            top_k=10,
        )

        # Build context and history strings
        local_context = self._build_context_block(chunks)

        return local_context

    async def get_global_document_context(self, msg_text: str) -> str:

        toc = await self._get_global_toc()

        relevant_toc_ids = []
        if len(toc) > 4:
            relevant_toc_ids = await self._get_relevant_Id(msg_text, toc)
        else:
            relevant_toc_ids = [row["TOCID"] for row in toc]

        # Stage 2 — Vector retrieval
        chunks = self._fetch_relevant_chunks(
            question=msg_text,
            toc_ids=relevant_toc_ids,
            country_id=None,
            pillar_id=None,
            top_k=10,
        )

        # Build context and history strings
        local_context = self._build_context_block(chunks)

        return local_context

    async def send_question_to_llm(
        self,
        questionText: str,
        ai_context: str,
        countryName: str,
        pillar_name: str,
        historyText: Optional[str] = None,
    ) -> str:

       return await self._synthesize_chat_answer(
            questionText, ai_context, countryName, pillar_name, historyText
        )
    
    async def send_cross_comparision_question_to_llm(
        self,
        questionText: str,
        ai_context: str,
        countryName: str,
        pillar_name: str,
        historyText: Optional[str] = None,
    ) -> str:

       return await self._synthesize_chat_answer(
            questionText, ai_context, countryName, pillar_name, historyText
        )

    async def _synthesize_chat_answer(
        self,
        questionText: str,
        ai_context: Any,
        countryName: str,
        pillar_name: str,
        historyText: Optional[str] = None,
    ) -> str:
        # context_text = self._context_to_text(ai_context)
        system_prompt = HSPromptTemplates.chat_system_prompt()
        user_prompt = HSPromptTemplates.chat_answer_user_prompt(
            ai_context, historyText, questionText, countryName, pillar_name
        )

        if question_needs_web_search(questionText, ai_context) and web_search_available():
            try:
                answer, web_sources = await invoke_with_web_search(
                    system_prompt,
                    user_prompt,
                    model=None,
                )
                return apply_verified_citations(
                    answer, merge_verified_sources(web_sources)
                )
            except Exception as exc:
                logger.warning(
                    "Web Search unavailable; falling back to RAG-only: %s", exc
                )

        answer = await self._llm_svc.invoke_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            label=f"rag_answer|program{countryName}|pillar{pillar_name}|q={questionText[:40]}",
        )
     
        return apply_verified_citations(answer,[])

    # @staticmethod
    # def _context_to_text(ai_context: Any) -> str:
    #     if ai_context is None:
    #         return ""
    #     if isinstance(ai_context, str):
    #         return ai_context
    #     if isinstance(ai_context, dict):
    #         return "\n".join(f"{k}: {value}" for k, value in ai_context.items())
    #     if isinstance(ai_context, (list, tuple, set)):
    #         lines = []
    #         for item in ai_context:
    #             if isinstance(item, dict):
    #                 lines.append(" | ".join(f"{k}: {v}" for k, v in item.items()))
    #             else:
    #                 lines.append(str(item))
    #         return "\n".join(lines)
    #     return str(ai_context)
    
    # ------------------------------------------------------------------ #
    #  Stage 1 — DB: fetch TOC                                           #
    #  ⚡ Tenant migration point: only this method touches the DB        #
    # ------------------------------------------------------------------ #

    async def _get_country_toc(
        self,
        country_id: int,
        pillar_id: Optional[int] = None,
    ) -> List[Dict]:
        """
        Fetch the Table-of-Contents entries for a country's uploaded documents.

        Returns a list of dicts with keys:
            TOCID, SectionPath, SectionTitle, SectionLevel, PillarID, FileName
        """
        query = """
            SELECT t.TOCID, t.SectionPath, t.SectionTitle, t.SectionLevel,
                   t.PillarID, cd.FileName
            FROM DocumentTOC t
            JOIN CountryDocuments cd ON cd.CountryDocumentID = t.CountryDocumentID
            WHERE t.CountryID = ? AND cd.IsDeleted = 0
        """
        # Future: add   AND t.TenantID = ?   when multi-tenant
        return await self._db.engine.fetch_dicts_async(query, (country_id,))

    async def _get_global_toc(self) -> List[Dict]:

        query = """
            SELECT t.TOCID, t.SectionPath, t.SectionTitle, t.SectionLevel,
                   t.PillarID, cd.FileName
            FROM DocumentTOC t
            JOIN CountryDocuments cd ON cd.CountryDocumentID = t.CountryDocumentID
            WHERE  cd.IsDeleted = 0 or DocumentLevel Like ?
        """
        documentLevel = "Global"

        return await self._db.engine.fetch_dicts_async(query, (documentLevel))

    # ------------------------------------------------------------------ #
    #  Stage 1 — LLM: route question to relevant TOC sections            #
    # ------------------------------------------------------------------ #

    async def _get_relevant_Id(
        self,
        question: str,
        toc: List[Dict],
    ) -> List[int]:
        """
        Ask the LLM which TOC section IDs are most relevant to the question.
        Returns a list of TOCID integers (may be empty).
        """
        if not toc:
            return []

        toc_text = "\n".join(
            f"[{row['TOCID']}] (Level {row['SectionLevel']}) {row['SectionPath']}"
            for row in toc
        )
        prompt = HSPromptTemplates.get_relevant_Id_prompt(toc_text, question)
        raw = await self._llm_svc.invoke_raw(
            prompt, label=f"rag_routing|q={question[:40]}"
        )

        match = re.search(r"\[[\d,\s]*\]", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []

    # ------------------------------------------------------------------ #
    #  Stage 2 — ChromaDB: vector search within sections                 #
    # ------------------------------------------------------------------ #

    def _fetch_relevant_chunks(
        self,
        question: str,
        toc_ids: List[int],
        country_id: Optional[int] = None,
        pillar_id: Optional[int] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Run a vector similarity search against the ChromaDB collection and
        return the top-k chunks, optionally filtered to the routed TOC IDs.
        """
        collection_name = (
            "Global"
            if country_id is None
            else (
                f"Country_{country_id}"
                if pillar_id is None
                else f"Country_Pillar_{country_id}"
            )
        )
        try:
            #    collections = self.client.list_collections()

            collection = self.client.get_collection(
                name=collection_name, embedding_function=self.embed_fn
            )
        except Exception as e:
            logger.error(f"Error fetching collection {collection_name}: {e}")
            return []

        where_filter = {"toc_id": {"$in": toc_ids}} if toc_ids else None
        results = collection.query(
            query_texts=[question],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "text": doc,
                    "section": meta.get("section_path", ""),
                    "file": meta.get("section_title", "") or meta.get("title", ""),
                    "relevance": round(1 - dist, 3),
                    "source_url": meta.get("source_url") or meta.get("sourceUrl") or "",
                    "source_name": meta.get("source_name") or meta.get("sourceName") or "",
                    "published_date": meta.get("published_date") or meta.get("source_data_year") or ""
                }
            )
        return chunks

    async def get_related_FAQ_IDs(
        self,
        question: str,
        toc: List[Dict],
    ) -> List[int]:
        """
        Ask the LLM which FAQ section IDs are most relevant to the question.
        Returns a list of FAQIDs integers (may be empty).
        """
        if not toc:
            return []

        toc_text = "\n".join(
            f"[{row['FAQID']}] (QuestionText {row['QuestionText']}) {row['Category']}"
            for row in toc
        )
        prompt = HSPromptTemplates.get_relevant_faqId_prompt(toc_text, question)
        raw = await self._llm_svc.invoke_raw(
            prompt, label=f"rag_routing|q={question[:80]}"
        )

        match = re.search(r"\[[\d,\s]*\]", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []


    async def country_executive_slides( self,  country_name: str, ai_country_context: str, allPillarContexts: str, year: int = None) -> Dict[str, Any]:

        try:

            # ---------------------------------------------------------
            # SYSTEM PROMPT
            # ---------------------------------------------------------
            system_prompt = (
                HSPromptTemplates.Country_executive_slides_prompt(
                    publicContext=ai_country_context,
                    allPillarContexts=allPillarContexts
                )
            )

            # ---------------------------------------------------------
            # USER TEMPLATE
            # ---------------------------------------------------------
            user_template = """
            country:
            {country_name}

            Year:
            {year}
            """

            # ---------------------------------------------------------
            # LLM CALL
            # ---------------------------------------------------------
            raw = await self._llm_svc.invoke_chain(
                system_prompt=system_prompt,
                user_template=user_template,
                variables={
                    "country_name": country_name,
                    "year": year
                },
                label=f"country-executive-slides|{country_name}",
            )

            analysis = json.loads(
                jrp.clean_json_response(raw)
            )

            return {
                "success": True,
                "data": analysis
            }

        except Exception as exc:
            logger.exception(
                "country_executive_slides failed"
            )

            return {
                "success": False,
                "error": str(exc)
            }


    @staticmethod
    def _next_news_pair(
        rows: List[Dict[str, str]],
        topics: tuple,
        start: int,
        blocked_countries: set,
        blocked_topics: set,
    ) -> Optional[tuple]:
        """Next unused country and topic, skipping codes the news API already rejected."""
        span = max(len(rows), len(topics))
        for step in range(span):
            idx = start + step
            row = rows[idx % len(rows)]
            topic = topics[idx % len(topics)]
            code = row["code"]
            if code in blocked_countries or is_invalid_country(code):
                continue
            if topic in blocked_topics:
                continue
            return row, topic
        return None

    async def _fetch_one_news_article(
        self,
        client: httpx.AsyncClient,
        row: Dict[str, str],
        topic: str,
    ) -> Optional[Dict[str, Any]]:
        """One list call, then one details call for the latest UUID. No URL retry."""
        try:
            return await fetch_first_article(
                client,
                row["code"],
                topic,
                country_name=row["name"],
                region=row["region"],
            )
        except FreeNewsRequestError as exc:
            if exc.invalid_country:
                note_invalid_country(exc.invalid_country)
            logger.warning(
                "Free News fetch failed for %s / %s: %s",
                row["code"],
                topic,
                exc,
            )
            # Rate limit or server error: do not send another request in this call.
            if exc.status_code == 429 or (exc.status_code or 0) >= 500:
                raise
            return None
        except Exception as exc:
            logger.warning(
                "Free News fetch failed for %s / %s: %s",
                row["code"],
                topic,
                exc,
            )
            return None

    async def _fetch_emerging_articles(
        self,
        max_records: int,
        query_variant: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        One Free News search per FastAPI call.

        Uses one Countries.CountryCode and one topic, then the latest UUID
        from that list. If that pair returns nothing, try one different
        country and one different topic, then stop.
        """
        del max_records  # countryCount stays on the public route; this call returns one article
        countries = await self._db.get_active_countries()
        rows = HSPromptTemplates.freenews_country_rows(countries)
        if not rows:
            raise ValueError("No country codes configured in Countries")

        topics = HSPromptTemplates.FREENEWS_TOPICS
        start = (
            int(query_variant)
            if query_variant is not None
            else HSPromptTemplates.pick_emerging_news_start_index()
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            first = self._next_news_pair(rows, topics, start, set(), set())
            if first is None:
                raise ValueError("No supported country codes available")

            row, topic = first
            article = await self._fetch_one_news_article(client, row, topic)
            if article:
                return [article]

            logger.info(
                "Free News returned nothing for %s / %s; trying one other country and topic",
                row["code"],
                topic,
            )
            fallback = self._next_news_pair(
                rows,
                topics,
                start + 1,
                {row["code"]},
                {topic},
            )
            if fallback is None:
                raise ValueError("Free News API returned no articles")

            fallback_row, fallback_topic = fallback
            article = await self._fetch_one_news_article(
                client, fallback_row, fallback_topic
            )
            if article:
                return [article]

        raise ValueError("Free News API returned no articles")

    async def emerging_trends_and_issues(
        self,
        country_count: int = 8,
        query_variant: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            max_records = max(1, min(250, country_count))

            now_utc = datetime.now(timezone.utc)

            # ---------------------------------------------------------
            # One country + one topic per Free News search; first UUID only
            # ---------------------------------------------------------
            articles_raw = await self._fetch_emerging_articles(
                max_records, query_variant=query_variant
            )
            articles: List[Dict[str, Any]] = []
            article_meta: Dict[str, Dict[str, str]] = {}
            for a in articles_raw[:max_records]:
                if not isinstance(a, dict):
                    continue
                url = str(a.get("url", "")).strip()
                title = str(a.get("title", "")).strip()
                if not url.startswith(("http://", "https://")) or not title:
                    continue

                article = {
                    "url": url,
                    "title": title,
                    "seendate": str(a.get("seendate", "")).strip(),
                    "language": str(a.get("language", "")).strip(),
                    "sourcecountry": str(a.get("sourcecountry", "")).strip(),
                    "countryName": str(a.get("countryName", "")).strip(),
                    "region": str(a.get("region", "")).strip(),
                    "topic": str(a.get("topic", "")).strip(),
                    "excerpt": str(a.get("excerpt", "")).strip(),
                }
                articles.append(article)
                article_meta[url] = article

            if not articles:
                raise ValueError("Insufficient usable news articles")

            system_prompt = HSPromptTemplates.emerging_trend_risk_prompt()
            user_template = HSPromptTemplates.emerging_trends_and_issues_user_prompt()

            raw = await self._llm_svc.invoke_chain(
                system_prompt=system_prompt,
                user_template=user_template,
                variables={
                    "current_date": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "articles_json": json.dumps(articles, ensure_ascii=False),
                },
                label="emerging-trends-and-issues",
            )

            analysis = json.loads(jrp.clean_json_response(raw))

            # Guardrail: cards keep the fetched URL, title, and Countries-table identity.
            cards = analysis.get("countries") or []
            if isinstance(cards, list):
                cleaned_cards: List[Dict[str, Any]] = []
                for c in cards:
                    if not isinstance(c, dict):
                        continue
                    u = str(c.get("sourceUrl", "")).strip()
                    meta = article_meta.get(u)
                    if not meta:
                        continue
                    c["title"] = meta["title"]
                    c["sourceUrl"] = meta["url"]
                    if meta.get("countryName"):
                        c["country"] = meta["countryName"]
                    if meta.get("sourcecountry"):
                        c["countryCode"] = meta["sourcecountry"]
                    if meta.get("region"):
                        c["region"] = meta["region"]
                    cleaned_cards.append(c)
                analysis["countries"] = cleaned_cards

            if not analysis.get("updatedAt"):
                analysis["updatedAt"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

            return {
                "success": True,
                "data": analysis,
            }

        except Exception as exc:
            logger.exception("emerging_trends_and_issues failed")

            return {
                "success": False,
                "error": str(exc),
            }


    async def pillar_live_signals(
        self,
        pillars: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            pillar_ids = sorted(pillars.keys())
            pillar_count = len(pillar_ids)
            id_range = (
                f"{pillar_ids[0]} through {pillar_ids[-1]}"
                if pillar_count > 1
                else str(pillar_ids[0]) if pillar_ids else "none"
            )
            countries = await self._db.get_active_countries()
            if not HSPromptTemplates.selected_country_names(countries):
                raise ValueError("No selected countries configured")
            selected_countries = HSPromptTemplates.format_selected_countries(countries)
            system_prompt = HSPillarPrompts.pillar_live_signals_prompt(
                pillars, selected_countries
            )

            user_template = f"""
            Generate the LIVE HS pillar signals feed (all {pillar_count} active pillars).

            Current UTC datetime (now):
            {{current_date}}

            Live coverage window start (48 hours before now):
            {{recency_cutoff}}

            Countries: {{selected_countries}}
            Name these countries. Do not say Africa, the continent, or Horn of Africa instead.

            Requirements:
            - Exactly {pillar_count} entries: pillarId {id_range}, each once.
            - Search each pillar domain before writing its card.
            - Use verified sourceUrl rules from the system prompt.
            """

            now_utc = datetime.now(timezone.utc)
            raw = await self._llm_svc.invoke_chain(
                system_prompt=system_prompt,
                user_template=user_template,
                variables={
                    "current_date": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "recency_cutoff": (now_utc - timedelta(hours=48)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "selected_countries": selected_countries,
                },
                label="pillar-live-signals",
            )

            analysis = json.loads(jrp.clean_json_response(raw))

            return {
                "success": True,
                "data": analysis,
            }

        except Exception as exc:
            logger.exception("pillar_live_signals failed")

            return {
                "success": False,
                "error": str(exc),
            }

    async def pillar_overview(self, pillars: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        try:
            countries = await self._db.get_active_countries()
            if not HSPromptTemplates.selected_country_names(countries):
                raise ValueError("No selected countries configured")
            selected_countries = HSPromptTemplates.format_selected_countries(countries)
            now_utc = datetime.now(timezone.utc)
            raw = await self._llm_svc.invoke_chain(
                system_prompt = HSPillarPrompts.pillar_overview_prompt(
                    pillars, selected_countries
                ),
                user_template = HSPillarPrompts.pillar_overview_user_prompt(),
                variables = {
                    "current_date": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "selected_countries": selected_countries,
                },
                label="pillar-overview",
                max_tokens=16000,
            )

            analysis = json.loads(jrp.clean_json_response(raw))
            if not analysis.get("updatedAt"):
                analysis["updatedAt"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

            return {
                "success": True,
                "data": analysis,
            }

        except Exception as exc:
            logger.exception("pillar_overview failed")
            return {
                "success": False,
                "error": str(exc),
            }


    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_context_block(chunks: List[Dict]) -> str:
        if not chunks:
           return ""

        from app.services.common.url_verifier import is_valid_source_url

        catalog: List[str] = []
        lines = ["=== FROM UPLOADED COUNTRY DOCUMENTS ==="]
        source_idx = 0
        for chunk in chunks:
            url = str(chunk.get("source_url") or "").strip()
            prefix = ""
            if is_valid_source_url(url):
                source_idx += 1
                name = chunk.get("source_name") or chunk.get("file") or "RAG"
                title = chunk.get("file") or chunk.get("section") or ""
                catalog.append(f"source_{source_idx} | {name} | {title} | {url}")
                prefix = f"[source_{source_idx}] "
            lines.append(f"{prefix}[{chunk['section']}]\n{chunk['text']}\n")

        if catalog:
            lines.insert(
                0,
                "=== VERIFIED RAG SOURCES (copy source_id only; never invent URLs) ===\n"
                + "\n".join(catalog)
                + "\n=== END VERIFIED RAG SOURCES ===\n",
            )
        return "\n".join(lines)
    
    @staticmethod
    def _build_history_str(chat_history: Optional[List[Dict]]) -> str:
        if not chat_history:
            return ""
        lines = []
        for msg in chat_history[-6:]:  # last 3 turns (user + assistant × 3)
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)


rag_query_service = RAGQueryService()
