"""
BOAMP (Bulletin Officiel des Annonces de Marchés Publics) — French public procurement.

Queries the BOAMP open-data API hosted on Opendatasoft for active tender
notices (avis de marchés) relevant to EHS / SDS / chemical safety.

Endpoint:
    GET https://boamp-datadila.opendatasoft.com/api/explore/v2.1/catalog/datasets/boamp/records

Free, public, no authentication required.  Returns JSON with a ``results``
array; each record contains BOAMP fields such as ``objet``, ``descripteurs``,
``organisme``, ``dateparution``, ``datelimitereponse``, ``idweb``, etc.

Usage:
    searcher = BoampSearcher()
    leads = searcher.search("sécurité chimique")
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

BOAMP_API_URL = (
    "https://boamp-datadila.opendatasoft.com/api/explore/v2.1"
    "/catalog/datasets/boamp/records"
)

# ---------------------------------------------------------------------------
# Keyword filtering — EHS / SDS domain (English + French)
# ---------------------------------------------------------------------------

# Strong keywords: if ANY of these appear, it's almost certainly relevant
STRONG_KEYWORDS: list[str] = [
    # English
    "safety data sheet", "sds management", "sds authoring",
    "msds", "chemical safety", "chemical management",
    "chemical inventory", "chemical compliance",
    "ghs", "globally harmonized",
    "ehs software", "ehs management", "ehs compliance",
    "hazardous material", "hazardous chemical", "hazardous substance",
    "hazard communication",
    "reach regulation", "clp regulation",
    "workplace safety software", "occupational safety software",
    "environmental health and safety",
    # French
    "fiche de données de sécurité", "fds",
    "sécurité chimique", "substances dangereuses",
    "reach", "clp", "risque chimique",
]

# Partial keywords: need 2+ to count
PARTIAL_KEYWORDS: list[str] = [
    # English
    "safety", "compliance", "environmental", "regulation",
    "chemical", "hazard", "risk management", "software platform",
    "occupational health", "dangerous goods", "toxic",
    # French
    "sécurité", "conformité", "environnement", "réglementation",
    "chimique", "dangereux", "risque", "logiciel",
    "santé au travail", "marchandises dangereuses", "toxique",
]


@dataclass
class BoampTenderLead:
    """A tender discovered from the French BOAMP portal."""

    lead_id: str
    title: str
    description: str
    agency: str
    source_portal: str = "boamp"
    source_url: str = ""
    submission_deadline: str = ""
    posted_date: str = ""
    value_amount: float = 0.0
    value_currency: str = "EUR"
    relevance_score: float = 0.0
    relevance_keywords: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


def score_relevance_boamp(title: str, description: str) -> tuple[float, list[str]]:
    """Score how relevant a BOAMP tender is to our EHS/SDS domain.

    Checks title + description against keyword lists.

    Returns:
        Tuple of (score 0.0–1.0, list of matched keywords).
    """
    text = f"{title} {description}".lower()
    matched: list[str] = []

    # Strong keyword matches (0.20 each, capped at 0.80)
    strong_count = 0
    for kw in STRONG_KEYWORDS:
        if kw in text:
            matched.append(kw)
            strong_count += 1

    # Partial keyword matches (0.05 each, capped at 0.20)
    partial_count = 0
    for kw in PARTIAL_KEYWORDS:
        if kw in text and kw not in matched:
            matched.append(kw)
            partial_count += 1

    strong_score = min(strong_count * 0.20, 0.80)
    partial_score = min(partial_count * 0.05, 0.20)
    total = min(strong_score + partial_score, 1.0)

    return round(total, 2), matched


class BoampSearcher:
    """Searches the French BOAMP portal for active tender notices.

    Queries the Opendatasoft-hosted BOAMP dataset, filters by
    EHS/SDS relevance keywords, and returns scored results.

    Usage:
        searcher = BoampSearcher()
        leads = searcher.search("sécurité chimique")
    """

    def __init__(self, timeout: float = 30.0, min_relevance: float = 0.10) -> None:
        self.timeout = timeout
        self.min_relevance = min_relevance
        logger.info("boamp_searcher_initialized")

    def search(
        self,
        user_query: str = "",
        max_results: int = 15,
        days_back: int = 60,
    ) -> list[BoampTenderLead]:
        """Search BOAMP for active EHS/SDS tenders.

        Args:
            user_query: User's search text (informational, not sent to API).
            max_results: Maximum results to return.
            days_back: How many days back to search.

        Returns:
            List of BoampTenderLead objects, filtered and sorted by relevance.
        """
        logger.info("boamp_search_start", days_back=days_back, max_results=max_results)

        # Build the where clause: active tenders published within date range.
        # BOAMP renamed/replaced its `nature` taxonomy in 2025; the old
        # AVIS_DE_MARCHE value no longer exists. Live values today are:
        # APPEL_OFFRE (call for tenders), PRE-INFORMATION, QUALIFICATION,
        # INTENTION_CONCLURE — we accept all four since the SDS scorer
        # filters anyway. We deliberately skip ATTRIBUTION (awarded),
        # ANNULATION (cancelled), MODIFICATION (amendment), and the
        # other administrative natures.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        where = (
            'nature IN ("APPEL_OFFRE","PRE-INFORMATION","QUALIFICATION","INTENTION_CONCLURE") '
            f'AND dateparution>="{cutoff}"'
        )

        # Fetch up to 2 pages (200 results max)
        all_records: list[dict] = []
        for page_idx in range(2):
            offset = page_idx * 100
            page_records = self._fetch_page(where, offset=offset, limit=100)
            all_records.extend(page_records)
            if len(page_records) < 100:
                break  # Last page

        logger.info("boamp_raw_results", total=len(all_records))

        # Parse, score, and filter
        leads: list[BoampTenderLead] = []
        seen_ids: set[str] = set()
        today = datetime.now(timezone.utc).date()

        for record in all_records:
            try:
                lead = self._parse_record(record)
                if not lead:
                    continue

                # Deduplicate
                if lead.lead_id in seen_ids:
                    continue
                seen_ids.add(lead.lead_id)

                # Skip expired tenders
                if lead.submission_deadline:
                    try:
                        dl = datetime.strptime(lead.submission_deadline, "%Y-%m-%d").date()
                        if dl < today:
                            continue
                    except ValueError:
                        pass

                # Score relevance
                score, keywords = score_relevance_boamp(lead.title, lead.description)
                lead.relevance_score = score
                lead.relevance_keywords = keywords

                if score >= self.min_relevance:
                    leads.append(lead)

            except Exception as exc:
                logger.debug("boamp_parse_error", error=str(exc))

        # Sort by relevance descending and cap
        leads.sort(key=lambda l: l.relevance_score, reverse=True)
        leads = leads[:max_results]

        logger.info(
            "boamp_search_complete",
            total_results=len(leads),
            top_score=leads[0].relevance_score if leads else 0,
        )

        return leads

    def _fetch_page(
        self,
        where: str,
        offset: int,
        limit: int,
    ) -> list[dict]:
        """Fetch a single page of records from the BOAMP API.

        Args:
            where: Opendatasoft query-language filter string.
            offset: Pagination offset.
            limit: Page size (max 100).

        Returns:
            List of raw record dicts from the ``results`` array.
        """
        params: dict[str, str | int] = {
            "where": where,
            "order_by": "dateparution desc",
            "limit": limit,
            "offset": offset,
            # Field rename in BOAMP schema (verified May 2026):
            #   "descripteurs"  →  "descripteur_libelle"  (human-readable
            #                       descriptor list; "descripteur_code"
            #                       is the machine-readable counterpart)
            #   "organisme"     →  "nomacheteur"
            # The old names now produce ODSQLError: Unknown field. Both
            # new fields exist on every record; we use the readable one
            # so keyword matching still scores correctly.
            "select": (
                "objet,descripteur_libelle,nomacheteur,dateparution,"
                "datelimitereponse,datefindiffusion,idweb,nature"
            ),
        }

        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(BOAMP_API_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

            records = data.get("results", [])
            logger.debug("boamp_page_fetched", offset=offset, count=len(records))
            return records

        except httpx.HTTPStatusError as exc:
            logger.error(
                "boamp_http_error",
                status=exc.response.status_code,
                detail=exc.response.text[:300],
            )
        except httpx.RequestError as exc:
            logger.error("boamp_request_error", error=str(exc))
        except Exception as exc:
            logger.error("boamp_fetch_error", error=str(exc))

        return []

    def _parse_record(self, record: dict) -> BoampTenderLead | None:
        """Parse a single BOAMP record into a BoampTenderLead.

        BOAMP record fields (may be at top level or nested in a ``fields`` dict):
            - ``objet``: tender title / subject
            - ``descripteurs``: description / keyword tags
            - ``organisme``: contracting authority / buyer name
            - ``datelimitereponse``: submission deadline
            - ``datefindiffusion``: end-of-publication date (fallback deadline)
            - ``dateparution``: publication date
            - ``idweb``: unique web identifier used in the public URL
            - ``nature``: notice type (e.g. AVIS_DE_MARCHE)
        """
        # Handle both flat and nested layouts
        fields = record.get("fields", record) if isinstance(record.get("fields"), dict) else record

        title = fields.get("objet", "") or ""
        if not title:
            return None

        # BOAMP renamed `descripteurs` -> `descripteur_libelle` and
        # `organisme` -> `nomacheteur` in 2025-2026.  Read the new names
        # but fall back to the old ones in case the API ever back-ports.
        descripteurs = (
            fields.get("descripteur_libelle")
            or fields.get("descripteurs")
            or ""
        )
        if isinstance(descripteurs, list):
            descripteurs = " ".join(str(d) for d in descripteurs)

        agency = (
            fields.get("nomacheteur")
            or fields.get("organisme")
            or ""
        )
        if not agency:
            agency = "Organisme public français"

        # Deadline: prefer datelimitereponse, fall back to datefindiffusion
        deadline_raw = fields.get("datelimitereponse", "") or fields.get("datefindiffusion", "") or ""
        deadline_iso = self._parse_date(str(deadline_raw))

        posted_raw = fields.get("dateparution", "") or ""
        posted_iso = self._parse_date(str(posted_raw))

        idweb = fields.get("idweb", "") or ""
        source_url = f"https://www.boamp.fr/avis/detail/{idweb}" if idweb else ""

        lead_id = str(idweb) if idweb else f"BOAMP-{uuid.uuid4().hex[:8].upper()}"

        return BoampTenderLead(
            lead_id=lead_id,
            title=title,
            description=descripteurs[:500] if descripteurs else "",
            agency=agency,
            source_portal="boamp",
            source_url=source_url,
            submission_deadline=deadline_iso,
            posted_date=posted_iso,
            value_amount=0.0,
            value_currency="EUR",
            relevance_keywords=[],
            raw_data={
                "idweb": idweb,
                "nature": fields.get("nature", ""),
                "has_strong_match": True,   # Updated after scoring
                "has_tender_signal": True,  # BOAMP = always a tender
                "is_empty_page": False,
            },
        )

    @staticmethod
    def _parse_date(date_str: str) -> str:
        """Parse BOAMP date strings to ISO YYYY-MM-DD.

        BOAMP dates may appear as ``YYYY-MM-DD``, ``YYYY-MM-DDThh:mm:ss``,
        ``DD/MM/YYYY``, or Unix-epoch timestamps.
        """
        if not date_str:
            return ""

        date_str = date_str.strip()

        # Already ISO date
        if re.match(r"^\d{4}-\d{2}-\d{2}", date_str):
            return date_str[:10]

        # French format DD/MM/YYYY
        m = re.match(r"^(\d{2})/(\d{2})/(\d{4})", date_str)
        if m:
            return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

        # Attempt generic parse
        try:
            from dateutil import parser as dateparser

            parsed = dateparser.parse(date_str, dayfirst=True)
            if parsed:
                return parsed.strftime("%Y-%m-%d")
        except Exception:
            pass

        return ""
