"""
Free News API client for the emerging-trends feed.

One search uses one country code and one topic. The latest article UUID
from that response is loaded once from the details endpoint.
The same URL is never retried.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import httpx

from app.config import settings
from app.services.common.country_prompt import HSPromptTemplates

logger = logging.getLogger(__name__)

FREENEWS_CACHE_TTL_SEC = 300

_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_invalid_countries: set[str] = set()


class FreeNewsRequestError(Exception):
    """One Free News call failed. Callers switch country and topic; they do not retry the URL."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        invalid_country: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.invalid_country = invalid_country


def is_invalid_country(country_code: str) -> bool:
    return country_code.strip().upper() in _invalid_countries


def note_invalid_country(country_code: str) -> None:
    code = country_code.strip().upper()
    if len(code) == 2:
        _invalid_countries.add(code)


def _cache_get(cache_key: str) -> Optional[Dict[str, Any]]:
    entry = _cache.get(cache_key)
    if not entry:
        return None
    ts, article = entry
    if time.monotonic() - ts > FREENEWS_CACHE_TTL_SEC:
        _cache.pop(cache_key, None)
        return None
    return article


def _cache_set(cache_key: str, article: Dict[str, Any]) -> None:
    _cache[cache_key] = (time.monotonic(), article)


def _api_headers() -> Dict[str, str]:
    api_key = (settings.FREENEWS_API_KEY or "").strip()
    if not api_key:
        raise ValueError("FREENEWS_API_KEY is not configured")
    return {"x-api-key": api_key}


def _latest_listing_item(data: list) -> Optional[Dict[str, Any]]:
    """Latest article in the listing. order_by=recent already puts it first."""
    candidates = [
        item for item in data
        if isinstance(item, dict) and str(item.get("uuid") or "").strip()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: str(item.get("published_at") or ""))


def _normalize_detail(
    detail: Dict[str, Any],
    *,
    country_code: str,
    country_name: str,
    region: str,
    topic: str,
) -> Optional[Dict[str, Any]]:
    url = str(detail.get("original_url") or "").strip()
    title = str(detail.get("title") or "").strip()
    if not url.startswith(("http://", "https://")) or not title:
        return None

    languages = detail.get("languages") or []
    language = str(languages[0]).strip() if languages else "en"
    excerpt = str(detail.get("incipit") or detail.get("body") or "").strip()
    excerpt = " ".join(excerpt.split())
    if len(excerpt) > 1200:
        excerpt = excerpt[:1200].rstrip()

    return {
        "url": url,
        "title": title,
        "seendate": str(detail.get("published_at") or "").strip(),
        "domain": str(detail.get("publisher") or "").strip(),
        "language": language,
        "sourcecountry": country_code.upper(),
        "countryName": country_name,
        "region": region or "Africa",
        "topic": topic,
        "socialimage": str(detail.get("thumbnail") or "").strip(),
        "excerpt": excerpt,
        "uuid": str(detail.get("uuid") or "").strip(),
    }


async def _get_json(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    """Single GET. A 400 or any other failure is returned to the caller once."""
    resp = await client.get(url, headers=_api_headers())
    if resp.status_code >= 400:
        invalid_country = None
        detail = resp.text[:300]
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = str(body.get("error") or body.get("detail") or detail)
                if str(body.get("error") or "").strip().lower() == "invalid country":
                    invalid_country = str(body.get("value") or "").strip().upper()
        except Exception:
            pass
        logger.warning(
            "Free News API request failed (%s): %s",
            resp.status_code,
            detail,
        )
        raise FreeNewsRequestError(
            detail or f"Free News API HTTP {resp.status_code}",
            status_code=resp.status_code,
            invalid_country=invalid_country,
        )

    payload = resp.json()
    if not isinstance(payload, dict):
        raise FreeNewsRequestError("Free News API returned a non-object payload")
    return payload


async def fetch_first_article(
    client: httpx.AsyncClient,
    country_code: str,
    topic: str,
    *,
    country_name: str,
    region: str,
) -> Optional[Dict[str, Any]]:
    """
    Search once with one country and one topic, then load details for the latest UUID.
    """
    code = country_code.strip().upper()
    topic_value = HSPromptTemplates._canonical_news_topic(topic)
    published_after, _published_before = HSPromptTemplates.emerging_trends_news_window()
    cache_key = f"freenews:{code}:{topic_value.casefold()}:{published_after}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("Free News cache hit for %s / %s", code, topic_value)
        return cached

    listing = await _get_json(
        client,
        HSPromptTemplates.emerging_trends_news_list_url(code, topic_value),
    )
    data = listing.get("data")
    if not isinstance(data, list) or not data:
        logger.info(
            "Free News API returned no articles for country=%s topic=%s",
            code,
            topic_value,
        )
        return None

    latest = _latest_listing_item(data)
    if latest is None:
        return None

    article_uuid = str(latest.get("uuid") or "").strip()
    detail_payload = await _get_json(
        client,
        HSPromptTemplates.emerging_trends_news_detail_url(article_uuid),
    )
    detail = detail_payload.get("data")
    if not isinstance(detail, dict):
        return None

    article = _normalize_detail(
        detail,
        country_code=code,
        country_name=country_name,
        region=region,
        topic=topic_value,
    )
    if article is None:
        return None

    _cache_set(cache_key, article)
    return article
