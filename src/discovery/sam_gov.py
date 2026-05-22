"""
SAM.gov Opportunity Scraper — Discovers federal tenders from SAM.gov.

WHAT IS SAM.GOV:
SAM.gov (System for Award Management) is the US federal government's official
procurement portal. Every federal agency posts contract opportunities here.
It publishes thousands of opportunities per month across all industries.

HOW WE USE IT:
SAM.gov provides a free public API (requires registration for an API key) at
https://api.sam.gov/opportunities/v2/search. We query it with filters for:
- NAICS codes relevant to EHS/SDS software (541620, 511210, etc.)
- Keywords like "SDS", "safety data sheet", "chemical", "EHS"
- Recent postings (last 7 days by default)

The API returns JSON with opportunity metadata: title, description, deadline,
solicitation number, agency, NAICS code, and links to full documents.

NAICS CODES WE WATCH:
These North American Industry Classification System codes cover our market:
- 541620: Environmental Consulting Services
- 541690: Other Scientific and Technical Consulting Services
- 511210: Software Publishers
- 541512: Computer Systems Design Services
- 541519: Other Computer Related Services
- 562910: Environmental Remediation Services

KEYWORD FILTERING:
Not every opportunity in our NAICS codes is relevant. A "Software Publishers"
tender might be for accounting software. After fetching results, we filter
by domain-specific keywords to keep only EHS/chemical safety opportunities.

DRY-RUN MODE:
Returns realistic mock tender data that exercises the full pipeline. The mock
tenders include a mix of highly relevant, somewhat relevant, and irrelevant
opportunities — just like real SAM.gov results.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAM_API_BASE = "https://api.sam.gov/opportunities/v2/search"

# NAICS codes relevant to EHS / SDS / chemical-safety federal contracts.
# Expanded session-7 to cover chemical manufacturing, testing labs,
# hazardous-waste handling, lab supplies, and safety training in
# addition to the original software/consulting cluster.
RELEVANT_NAICS: list[str] = [
    # --- core scoring set (session-1) ---
    "541620",  # Environmental Consulting Services
    "541690",  # Other Scientific and Technical Consulting
    "511210",  # Software Publishers
    "541512",  # Computer Systems Design Services
    "541519",  # Other Computer Related Services
    "562910",  # Environmental Remediation Services
    # --- chemical manufacturing (3251xx / 3252xx / 3259xx) ---
    "325",     # Chemical mfg (broad prefix catch)
    "325180",  # Other Basic Inorganic Chemical Manufacturing
    "325199",  # All Other Basic Organic Chemical Manufacturing
    "325410",  # Pharmaceutical & Medicine Manufacturing
    "325510",  # Paint & Coating Manufacturing
    "325611",  # Soap & Other Detergent Manufacturing
    "325998",  # All Other Misc Chemical Product / Preparation
    # --- testing & analysis labs ---
    "541380",  # Testing Laboratories
    "541713",  # Research and Development in Nanotechnology
    "541714",  # R&D in Biotech (except nano)
    "541715",  # R&D in Physical/Engineering/Life Sciences
    # --- hazardous waste & remediation ---
    "562112",  # Hazardous Waste Collection
    "562211",  # Hazardous Waste Treatment & Disposal
    "562112",  # (dup intentional — kept for grep visibility)
    "562998",  # All Other Misc Waste Management Services
    # --- training in safety ---
    "611430",  # Professional & Management Development Training
    "611699",  # All Other Misc Schools & Instruction
]

# Keywords that indicate EHS/SDS/chemical safety relevance
RELEVANCE_KEYWORDS: list[str] = [
    "safety data sheet", "sds", "msds",
    "chemical safety", "chemical management", "chemical inventory",
    "ghs", "globally harmonized",
    "ehs", "environment health safety", "environmental health",
    "hazardous material", "hazardous chemical", "hazmat",
    "osha", "hcs", "hazard communication",
    "regulatory compliance", "compliance software",
    "sds management", "sds authoring",
    "tier ii", "epcra", "toxic release",
    "workplace safety", "occupational safety",
]

# Broader keywords that suggest partial relevance
PARTIAL_KEYWORDS: list[str] = [
    "safety", "compliance", "environmental", "regulation",
    "chemical", "hazard", "risk management", "software platform",
    "cloud-based", "saas",
]


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class TenderLead:
    """A discovered tender opportunity, pre-evaluation.

    This is lighter than the full TenderState — it's what the scraper
    produces. The Discover node (Step 14) converts these into full
    TenderState dicts to feed into the graph.

    Attributes:
        lead_id: Unique identifier (SAM.gov solicitation number or UUID).
        title: Opportunity title.
        description: Summary description.
        agency: Issuing agency name.
        source_portal: Always "sam.gov" for this scraper.
        source_url: Direct URL to the opportunity on SAM.gov.
        naics_code: Primary NAICS code.
        submission_deadline: Deadline as ISO string.
        posted_date: When the opportunity was posted.
        relevance_score: 0.0-1.0 based on keyword matching.
        relevance_keywords: Which keywords matched.
        raw_data: Full API response for this opportunity (for debugging).
    """
    lead_id: str
    title: str
    description: str
    agency: str
    source_portal: str = "sam.gov"
    source_url: str = ""
    naics_code: str = ""
    submission_deadline: str = ""
    posted_date: str = ""
    relevance_score: float = 0.0
    relevance_keywords: list[str] = field(default_factory=list)
    # Direct download URLs the SAM.gov API surfaced for this opportunity
    # (resourceLinks).  When non-empty, the bridge skips HTML scraping
    # and downloads these directly during pursuit-attachment fetch.
    attachment_urls: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lead_id": self.lead_id,
            "title": self.title,
            "description": self.description,
            "agency": self.agency,
            "source_portal": self.source_portal,
            "source_url": self.source_url,
            "naics_code": self.naics_code,
            "submission_deadline": self.submission_deadline,
            "posted_date": self.posted_date,
            "relevance_score": self.relevance_score,
            "relevance_keywords": self.relevance_keywords,
            "attachment_urls": self.attachment_urls,
        }


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def score_relevance(title: str, description: str) -> tuple[float, list[str]]:
    """Score how relevant an opportunity is to our EHS/SDS domain.

    Checks title and description against keyword lists. Primary keywords
    (strong EHS/SDS match) contribute more than partial keywords.

    Returns:
        Tuple of (score 0.0-1.0, list of matched keywords)
    """
    text = f"{title} {description}".lower()
    matched: list[str] = []

    # Strong matches (worth 0.15 each, capped at 0.75)
    strong_matches = 0
    for kw in RELEVANCE_KEYWORDS:
        if kw in text:
            matched.append(kw)
            strong_matches += 1

    # Partial matches (worth 0.05 each, capped at 0.25)
    partial_matches = 0
    for kw in PARTIAL_KEYWORDS:
        if kw in text and kw not in matched:
            matched.append(kw)
            partial_matches += 1

    strong_score = min(strong_matches * 0.15, 0.75)
    partial_score = min(partial_matches * 0.05, 0.25)
    total = min(strong_score + partial_score, 1.0)

    return round(total, 2), matched


# ---------------------------------------------------------------------------
# SAM.gov scraper class
# ---------------------------------------------------------------------------

class SamGovScraper:
    """Fetches and filters federal tender opportunities from SAM.gov.

    Usage:
        scraper = SamGovScraper()  # uses DRY_RUN from env
        leads = scraper.fetch_opportunities(days_back=7)
        for lead in leads:
            print(f"[{lead.relevance_score:.0%}] {lead.title}")

    Args:
        api_key: SAM.gov API key. If None, reads from SAM_GOV_API_KEY env var.
        dry_run: If True, return mock data. Default reads from DRY_RUN env var.
        min_relevance: Minimum relevance score to include. Default 0.1.
    """

    def __init__(
        self,
        api_key: str | None = None,
        dry_run: bool | None = None,
        min_relevance: float = 0.1,
    ) -> None:
        if dry_run is None:
            dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

        self.dry_run = dry_run
        self.min_relevance = min_relevance
        self._api_key = api_key or os.getenv("SAM_GOV_API_KEY", "")

        if not self.dry_run and not self._api_key:
            raise ValueError(
                "SAM_GOV_API_KEY is required when DRY_RUN is disabled. "
                "Register for a free key at https://sam.gov/content/entity-registration"
            )

        logger.info(
            "sam_gov_scraper_initialized",
            dry_run=self.dry_run,
            min_relevance=self.min_relevance,
        )

    def fetch_opportunities(
        self,
        days_back: int = 7,
        limit: int = 100,
        naics_codes: list[str] | None = None,
    ) -> list[TenderLead]:
        """Fetch recent opportunities from SAM.gov.

        Args:
            days_back: How many days back to search. Default 7.
            limit: Maximum number of raw results to fetch. Default 100.
            naics_codes: NAICS codes to filter by. Default uses RELEVANT_NAICS.

        Returns:
            List of TenderLead objects, filtered and scored for relevance,
            sorted by relevance_score descending.
        """
        codes = naics_codes or RELEVANT_NAICS

        logger.info(
            "fetching_opportunities",
            days_back=days_back,
            limit=limit,
            naics_codes=codes,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            raw_leads = self._fetch_dry_run()
        else:
            raw_leads = self._fetch_real(days_back, limit, codes)

        # Score and filter for relevance
        scored_leads: list[TenderLead] = []
        for lead in raw_leads:
            score, keywords = score_relevance(lead.title, lead.description)
            lead.relevance_score = score
            lead.relevance_keywords = keywords

            if score >= self.min_relevance:
                scored_leads.append(lead)

        # Sort by relevance (highest first)
        scored_leads.sort(key=lambda l: l.relevance_score, reverse=True)

        logger.info(
            "opportunities_fetched",
            raw_count=len(raw_leads),
            filtered_count=len(scored_leads),
            top_score=scored_leads[0].relevance_score if scored_leads else 0.0,
        )

        return scored_leads

    def fetch_and_deduplicate(
        self,
        known_ids: set[str],
        days_back: int = 7,
    ) -> list[TenderLead]:
        """Fetch opportunities and exclude already-known tender IDs.

        In production, `known_ids` comes from the database — all tender_ids
        that have already been processed. This prevents re-evaluating the
        same tender every time the scraper runs.

        Args:
            known_ids: Set of tender IDs already in the system.
            days_back: How many days back to search.

        Returns:
            Only NEW opportunities not in known_ids.
        """
        all_leads = self.fetch_opportunities(days_back=days_back)
        new_leads = [l for l in all_leads if l.lead_id not in known_ids]

        logger.info(
            "deduplication_complete",
            total=len(all_leads),
            known=len(all_leads) - len(new_leads),
            new=len(new_leads),
        )

        return new_leads

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _fetch_real(
        self,
        days_back: int,
        limit: int,
        naics_codes: list[str],
    ) -> list[TenderLead]:
        """Fetch from the real SAM.gov API.

        SAM.gov API docs: https://open.gsa.gov/api/get-opportunities-public-api/
        """
        import httpx

        posted_from = (
            datetime.now(timezone.utc) - timedelta(days=days_back)
        ).strftime("%m/%d/%Y")
        posted_to = datetime.now(timezone.utc).strftime("%m/%d/%Y")

        leads: list[TenderLead] = []
        rate_limited = False  # 429 is a DAILY quota; if we hit it once,
                              # every subsequent NAICS code will 429 too.
                              # Skip them instead of hammering the API.

        for naics in naics_codes:
            if rate_limited:
                break

            params = {
                "api_key": self._api_key,
                "postedFrom": posted_from,
                "postedTo": posted_to,
                "ncode": naics,
                "limit": min(limit, 100),
                "offset": 0,
            }

            try:
                response = httpx.get(
                    SAM_API_BASE,
                    params=params,
                    timeout=30.0,
                    follow_redirects=True,
                )
                if response.status_code == 429:
                    rate_limited = True
                    logger.warning(
                        "sam_gov_rate_limited",
                        msg="SAM.gov daily quota exhausted; skipping remaining NAICS codes",
                    )
                    break
                response.raise_for_status()
                data = response.json()

                for opp in data.get("opportunitiesData", []):
                    deadline = opp.get("responseDeadLine", "")
                    posted = opp.get("postedDate", "")
                    sol_number = opp.get("solicitationNumber", str(uuid.uuid4())[:12])

                    # SAM.gov exposes attachment URLs via `resourceLinks`
                    # (a list of direct file URLs).  When the operator
                    # later promotes this lead to a pursuit, the bridge
                    # picks these up via /api/tender-pursuits/<id>/source
                    # and downloads them directly — skipping the
                    # HTML-scraping fallback path entirely.
                    raw_links = opp.get("resourceLinks") or []
                    attachment_urls: list[str] = []
                    if isinstance(raw_links, list):
                        for link in raw_links:
                            if isinstance(link, str) and link.startswith("http"):
                                attachment_urls.append(link)
                            elif isinstance(link, dict):
                                # SAM.gov has been seen returning both
                                # bare strings and dicts shaped like
                                # {"url": "..."} depending on the
                                # opportunity type.  Tolerate both.
                                u = link.get("url") or link.get("href")
                                if isinstance(u, str) and u.startswith("http"):
                                    attachment_urls.append(u)

                    # SAM.gov requires the `/view` suffix on the
                    # opportunity URL.  Without it the public site 404s
                    # (the noticeId path alone is the API resource, not
                    # the rendered detail page).
                    notice_id = opp.get("noticeId", "")
                    source_url = (
                        f"https://sam.gov/opp/{notice_id}/view"
                        if notice_id
                        else "https://sam.gov/"
                    )

                    leads.append(TenderLead(
                        lead_id=sol_number,
                        title=opp.get("title", "Untitled"),
                        description=opp.get("description", "")[:2000],
                        agency=opp.get("fullParentPathName", "Unknown Agency"),
                        source_url=source_url,
                        naics_code=naics,
                        submission_deadline=deadline,
                        posted_date=posted,
                        attachment_urls=attachment_urls,
                        raw_data=opp,
                    ))

            except httpx.HTTPStatusError as exc:
                # Some 4xx codes are also terminal (e.g. 401 invalid key); skip
                # the rest if the API key isn't accepted.
                if exc.response.status_code in (401, 403):
                    rate_limited = True
                    logger.error(
                        "sam_gov_auth_failed",
                        status_code=exc.response.status_code,
                        msg="SAM.gov rejected API key; skipping remaining NAICS codes",
                    )
                    break
                logger.error("sam_gov_api_error", naics=naics, error=str(exc))
            except Exception as exc:
                logger.error("sam_gov_api_error", naics=naics, error=str(exc))

        return leads

    def _fetch_dry_run(self) -> list[TenderLead]:
        """Return realistic mock tender data for testing.

        Includes a mix of:
        - Highly relevant (SDS/EHS software)
        - Somewhat relevant (general environmental)
        - Irrelevant (included to test filtering)
        """
        now = datetime.now(timezone.utc)
        deadline = (now + timedelta(days=45)).isoformat()
        posted = now.isoformat()

        return [
            TenderLead(
                lead_id="SAM-2026-EHS-001",
                title="Cloud-Based SDS Management Platform for EPA Region 5",
                description=(
                    "The Environmental Protection Agency Region 5 seeks a cloud-based "
                    "Safety Data Sheet management system capable of GHS classification, "
                    "chemical inventory tracking, OSHA HCS compliance, and Tier II "
                    "regulatory reporting for 12 facilities across 6 states. Must include "
                    "mobile access, QR code scanning, and automated SDS revision tracking. "
                    "ISO 27001 or equivalent security certification required."
                ),
                agency="Environmental Protection Agency",
                source_url="https://sam.gov/opp/mock-ehs-001",
                naics_code="541620",
                submission_deadline=deadline,
                posted_date=posted,
            ),
            TenderLead(
                lead_id="SAM-2026-CHEM-002",
                title="Hazardous Chemical Inventory and Compliance Software",
                description=(
                    "Department of Defense installation requires hazardous chemical "
                    "inventory management software with EPCRA/CERCLA reporting, "
                    "safety data sheet distribution, and workplace hazard communication "
                    "program support. Cloud or on-premise deployment acceptable. "
                    "Must comply with DoD cybersecurity requirements."
                ),
                agency="Department of Defense",
                source_url="https://sam.gov/opp/mock-chem-002",
                naics_code="511210",
                submission_deadline=(now + timedelta(days=30)).isoformat(),
                posted_date=posted,
            ),
            TenderLead(
                lead_id="SAM-2026-ENV-003",
                title="Environmental Monitoring and Compliance Platform",
                description=(
                    "State environmental agency seeks a comprehensive environmental "
                    "compliance platform for air quality monitoring, water discharge "
                    "tracking, and waste management reporting. Must integrate with "
                    "existing LIMS systems and support regulatory submission workflows."
                ),
                agency="State Environmental Agency",
                source_url="https://sam.gov/opp/mock-env-003",
                naics_code="541620",
                submission_deadline=(now + timedelta(days=60)).isoformat(),
                posted_date=posted,
            ),
            TenderLead(
                lead_id="SAM-2026-IT-004",
                title="Enterprise Resource Planning System Modernization",
                description=(
                    "General Services Administration seeks a vendor for ERP system "
                    "modernization including financial management, human resources, "
                    "procurement, and asset tracking modules. Must be FedRAMP "
                    "authorized. Cloud-based SaaS deployment required."
                ),
                agency="General Services Administration",
                source_url="https://sam.gov/opp/mock-it-004",
                naics_code="541512",
                submission_deadline=(now + timedelta(days=90)).isoformat(),
                posted_date=posted,
            ),
            TenderLead(
                lead_id="SAM-2026-SAFE-005",
                title="Workplace Safety Training and Compliance Management System",
                description=(
                    "OSHA seeks a software platform for managing workplace safety "
                    "training records, incident reporting, and compliance tracking "
                    "across federal facilities. Must support hazard communication "
                    "training, chemical safety programs, and occupational safety "
                    "metrics dashboards."
                ),
                agency="Occupational Safety and Health Administration",
                source_url="https://sam.gov/opp/mock-safe-005",
                naics_code="541519",
                submission_deadline=(now + timedelta(days=40)).isoformat(),
                posted_date=posted,
            ),
        ]