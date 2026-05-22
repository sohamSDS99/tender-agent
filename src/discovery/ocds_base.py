"""
Shared OCDS (Open Contracting Data Standard) utilities.

Provides common keyword scoring, OCDS release parsing, and a base
TenderLead dataclass used by all OCDS-based country integrations.

Every government API that publishes OCDS data uses the same schema:
  - tender.title, tender.description
  - tender.tenderPeriod.endDate  (submission deadline)
  - buyer.name / parties[role=buyer].name
  - tender.classification.id  (CPV code)
  - tender.value.amount / tender.value.currency

This module centralises the parsing so country modules are thin wrappers
that only define their API endpoint and any country-specific keywords.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared EHS/SDS keyword lists — used by all country integrations
# ---------------------------------------------------------------------------

# Strong keywords: if ANY of these appear, it's almost certainly relevant
STRONG_KEYWORDS: list[str] = [
    "safety data sheet", "sds management", "sds authoring",
    "msds", "material safety data sheet",
    "chemical safety", "chemical management",
    "chemical inventory", "chemical compliance",
    "ghs", "globally harmonized",
    "ehs software", "ehs management", "ehs compliance",
    "hazardous material", "hazardous chemical", "hazardous substance",
    "hazard communication",
    "reach regulation", "clp regulation",
    "workplace safety software", "occupational safety software",
    "environmental health and safety",
]

# Partial keywords: need 2+ to count
PARTIAL_KEYWORDS: list[str] = [
    "safety", "compliance", "environmental", "regulation",
    "chemical", "hazard", "risk management", "software platform",
    "occupational health", "dangerous goods", "toxic",
    "waste management", "pollution", "contamination",
]

# CPV codes (EU Common Procurement Vocabulary) relevant to SDS / EHS /
# chemical-safety / lab work. Used by every OCDS-based source for the
# 0.15 CPV bonus in score_relevance().
#
# Expanded session-7 to surface more chemical-adjacent procurement:
# industrial chemicals, lab instruments, pollution monitoring, hazmat
# transport, safety training, and R&D services.
RELEVANT_CPV_PREFIXES: list[str] = [
    # --- core SDS / chemical safety ---
    "905",       # Environmental services (broad)
    "713172",    # Health and safety services
    "33141",     # Medical / lab chemicals (33141500 = serums etc.)
    "33342",     # Safety equipment (33342000)
    "33680",     # Pharmaceutical products + reagents
    "331410",    # Industrial chemicals
    "24",        # Chemical products (24*)  ← broadest catch
    # --- environmental & hazmat ---
    "9051",      # Refuse + waste services
    "9052",      # Hazardous / toxic / radioactive waste (90520000)
    "905240",    # Hazardous waste management (specific)
    "9073",      # Environmental monitoring / pollution measurement
    "606",       # Transport of hazardous goods (60600000-class)
    # --- lab + testing + analysis ---
    "38540",     # Laboratory instruments (38540000)
    "71621",     # Technical analysis / consultancy services
    "7300",      # R&D services
    # --- safety training + consulting ---
    "79417",     # Safety consultancy services
    "80511",     # Staff safety training
    # --- enabling tech ---
    "480000",    # Software packages
    "720000",    # IT services
]


@dataclass
class OcdsTenderLead:
    """A tender discovered from any OCDS-compliant government API."""
    lead_id: str
    title: str
    description: str
    agency: str
    source_portal: str = ""
    source_url: str = ""
    submission_deadline: str = ""
    posted_date: str = ""
    value_amount: float = 0.0
    value_currency: str = ""
    cpv_code: str = ""
    relevance_score: float = 0.0
    relevance_keywords: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


def score_relevance(
    title: str,
    description: str,
    cpv_code: str = "",
    extra_strong: list[str] | None = None,
    extra_partial: list[str] | None = None,
) -> tuple[float, list[str]]:
    """Score how relevant a tender is to our EHS/SDS domain.

    Uses the shared keyword lists plus optional country-specific extras.

    Args:
        title: Tender title.
        description: Tender description.
        cpv_code: CPV classification code (if available).
        extra_strong: Additional strong keywords for this country.
        extra_partial: Additional partial keywords for this country.

    Returns:
        (score, matched_keywords) tuple.
    """
    text = f"{title} {description}".lower()
    matched: list[str] = []

    all_strong = STRONG_KEYWORDS + (extra_strong or [])
    all_partial = PARTIAL_KEYWORDS + (extra_partial or [])

    # Strong keyword matches (0.20 each, capped at 0.80)
    strong_count = 0
    for kw in all_strong:
        if kw in text:
            matched.append(kw)
            strong_count += 1

    # Partial keyword matches (0.05 each, capped at 0.20)
    partial_count = 0
    for kw in all_partial:
        if kw in text and kw not in matched:
            matched.append(kw)
            partial_count += 1

    # CPV code bonus (0.15 if relevant)
    cpv_bonus = 0.0
    if cpv_code:
        for prefix in RELEVANT_CPV_PREFIXES:
            if cpv_code.startswith(prefix):
                cpv_bonus = 0.15
                matched.append(f"cpv:{cpv_code}")
                break

    strong_score = min(strong_count * 0.20, 0.80)
    partial_score = min(partial_count * 0.05, 0.20)
    total = min(strong_score + partial_score + cpv_bonus, 1.0)

    return round(total, 2), matched


def parse_ocds_release(
    release: dict,
    source_portal: str,
    build_url: Any = None,
) -> OcdsTenderLead | None:
    """Parse a single OCDS release into an OcdsTenderLead.

    Works with any OCDS 1.1 compliant release object.

    Args:
        release: Raw OCDS release dict.
        source_portal: Portal identifier (e.g. "austender", "sa_etender").
        build_url: Optional callable(release) -> str to build the source URL.

    Returns:
        OcdsTenderLead or None if the release is unparseable/expired.
    """
    ocid = release.get("ocid", "")
    release_id = release.get("id", "")

    tender = release.get("tender", {})
    if not tender:
        return None

    title = tender.get("title", "")
    description = tender.get("description", "")
    if not title:
        return None

    # Extract buyer name
    buyer = release.get("buyer", {})
    buyer_name = buyer.get("name", "")
    if not buyer_name:
        parties = release.get("parties", [])
        for party in parties:
            roles = party.get("roles", [])
            if "buyer" in roles:
                buyer_name = party.get("name", "")
                break
    if not buyer_name:
        buyer_name = "Government"

    # Reject tenders that haven't opened yet (status "planning" /
    # "planned") so operators never see forthcoming calls in their
    # Inbox. OCDS tender.status vocabulary:
    #   planning   — internal planning, not public
    #   planned    — published intent, no bids accepted yet
    #   active     — accepting bids (what we want)
    #   cancelled / unsuccessful / complete / withdrawn — terminal
    tender_status = (tender.get("status") or "").lower()
    if tender_status in ("planning", "planned"):
        return None

    # Extract deadline (tenderPeriod.endDate)
    tender_period = tender.get("tenderPeriod", {})
    deadline = tender_period.get("endDate", "")

    # Reject if the tender hasn't opened for bids yet — startDate
    # in the future is the OCDS-shape equivalent of EU F&T's
    # "forthcoming" status. Same operator-experience rule applies:
    # don't surface calls that aren't actually accepting submissions.
    start_date_raw = tender_period.get("startDate", "")
    if start_date_raw:
        start_iso = _parse_date(start_date_raw)
        if start_iso:
            try:
                sd = datetime.strptime(start_iso, "%Y-%m-%d").date()
                if sd > datetime.now(timezone.utc).date():
                    return None
            except Exception:
                pass

    # Posted date
    posted_date = release.get("date", "")

    # Value
    value = tender.get("value", {})
    value_amount = value.get("amount", 0) or 0
    value_currency = value.get("currency", "")

    # CPV code
    classification = tender.get("classification", {})
    cpv_code = classification.get("id", "")

    # Build source URL
    source_url = ""
    if build_url:
        try:
            source_url = build_url(release)
        except Exception:
            pass

    # Parse dates
    deadline_iso = _parse_date(deadline)
    posted_iso = _parse_date(posted_date)

    # Skip expired tenders
    if deadline_iso:
        try:
            dl = datetime.strptime(deadline_iso, "%Y-%m-%d").date()
            if dl < datetime.now(timezone.utc).date():
                return None
        except Exception:
            pass

    return OcdsTenderLead(
        lead_id=release_id or ocid or f"{source_portal}-{uuid.uuid4().hex[:8].upper()}",
        title=title,
        description=description[:500] if description else "",
        agency=buyer_name,
        source_portal=source_portal,
        source_url=source_url,
        submission_deadline=deadline_iso,
        posted_date=posted_iso,
        value_amount=float(value_amount),
        value_currency=value_currency,
        cpv_code=cpv_code,
        relevance_keywords=[],
        raw_data={
            "ocid": ocid,
            "source_portal": source_portal,
        },
    )


def _parse_date(date_str: str) -> str:
    """Parse OCDS date formats to ISO YYYY-MM-DD."""
    if not date_str:
        return ""

    date_str = str(date_str).strip()

    # Already ISO date
    if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
        return date_str[:10]

    # Try dateutil
    try:
        from dateutil import parser as dateparser
        parsed = dateparser.parse(date_str)
        if parsed:
            return parsed.strftime("%Y-%m-%d")
    except Exception:
        pass

    return ""
