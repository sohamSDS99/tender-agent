"""
Discovery Coordinator — Runs all scrapers and produces graph-ready TenderState dicts.

WHY A COORDINATOR:
The graph processes one tender at a time. But we have multiple discovery sources
(SAM.gov, email, potentially more in the future). The coordinator's job is to:

1. Run every scraper
2. Combine all results into a single list
3. Deduplicate across sources (same tender might appear on SAM.gov AND in email)
4. Sort by relevance (best opportunities first)
5. Convert each TenderLead into an initial TenderState dict ready for graph.invoke()

HOW IT FITS IN THE ARCHITECTURE:
In production, a cron job (every 6 hours) calls coordinator.discover_new_tenders().
For each tender returned, it calls graph.invoke(tender_state). The coordinator runs
OUTSIDE the graph — it's the trigger, not a node.

The Discover NODE inside the graph (Step 14, discover_node) handles the per-tender
processing: validation, enrichment, and status setting.

DEDUPLICATION:
A tender might appear on SAM.gov with solicitation number "W911NF-26-R-0042" AND
arrive via email with subject "RFP: EHS Software for DoD". These are the same
opportunity. We deduplicate by:
1. Exact ID match (if both sources use the same solicitation number)
2. Title similarity (fuzzy match for cross-source duplicates)
For now, we use exact ID dedup. Title-based fuzzy dedup is a future enhancement.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import structlog

from src.discovery.email_monitor import EmailMonitor
from src.discovery.sam_gov import SamGovScraper, TenderLead

logger = structlog.get_logger(__name__)


class DiscoveryCoordinator:
    """Runs all discovery sources and returns graph-ready tender states.

    Usage:
        coordinator = DiscoveryCoordinator()
        new_tenders = coordinator.discover_new_tenders(known_ids=set())
        for tender_state in new_tenders:
            result = graph.invoke(tender_state)

    Args:
        sam_scraper: SamGovScraper instance. Created automatically if None.
        email_monitor: EmailMonitor instance. Created automatically if None.
        min_relevance: Minimum relevance score to process. Default 0.15.
    """

    def __init__(
        self,
        sam_scraper: SamGovScraper | None = None,
        email_monitor: EmailMonitor | None = None,
        min_relevance: float = 0.15,
    ) -> None:
        self.sam_scraper = sam_scraper or SamGovScraper(min_relevance=min_relevance)
        self.email_monitor = email_monitor or EmailMonitor(min_relevance=min_relevance)
        self.min_relevance = min_relevance

        logger.info("discovery_coordinator_initialized")

    def discover_new_tenders(
        self,
        known_ids: set[str] | None = None,
        days_back: int = 7,
        max_tenders: int = 20,
    ) -> list[dict[str, Any]]:
        """Run all scrapers and return new tender states ready for the graph.

        Args:
            known_ids: Set of tender IDs already in the system (from DB).
                If None, no deduplication is done.
            days_back: How far back to search. Default 7 days.
            max_tenders: Maximum tenders to return per run. Default 20.
                Prevents the agent from being overwhelmed by too many
                tenders at once.

        Returns:
            List of initial TenderState dicts, sorted by relevance (highest first).
            Each dict is ready to pass to graph.invoke().
        """
        known = known_ids or set()

        logger.info(
            "discovery_run_start",
            known_count=len(known),
            days_back=days_back,
        )

        all_leads: list[TenderLead] = []

        # --- Source 1: SAM.gov ---
        try:
            sam_leads = self.sam_scraper.fetch_and_deduplicate(
                known_ids=known,
                days_back=days_back,
            )
            all_leads.extend(sam_leads)
            logger.info("sam_gov_leads", count=len(sam_leads))
        except Exception as exc:
            logger.error("sam_gov_scraper_failed", error=str(exc))

        # --- Source 2: Email ---
        try:
            email_leads = self.email_monitor.check_and_deduplicate(
                known_ids=known | {l.lead_id for l in all_leads},
            )
            all_leads.extend(email_leads)
            logger.info("email_leads", count=len(email_leads))
        except Exception as exc:
            logger.error("email_monitor_failed", error=str(exc))

        # --- Sort by relevance and cap ---
        all_leads.sort(key=lambda l: l.relevance_score, reverse=True)
        top_leads = all_leads[:max_tenders]

        # --- Convert to TenderState dicts ---
        tender_states = [_lead_to_tender_state(lead) for lead in top_leads]

        logger.info(
            "discovery_run_complete",
            total_leads=len(all_leads),
            returning=len(tender_states),
            top_score=top_leads[0].relevance_score if top_leads else 0.0,
        )

        return tender_states

    def discover_single(self, lead: TenderLead) -> dict[str, Any]:
        """Convert a single TenderLead into a graph-ready TenderState.

        Useful for manually adding a tender to the pipeline without
        running the full discovery cycle.

        Args:
            lead: A TenderLead from any source.

        Returns:
            Initial TenderState dict ready for graph.invoke().
        """
        return _lead_to_tender_state(lead)


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------

def _lead_to_tender_state(lead: TenderLead) -> dict[str, Any]:
    """Convert a TenderLead into an initial TenderState dict.

    This is the bridge between discovery and the graph. The TenderLead is a
    lightweight struct from the scraper; the TenderState is the full state
    dict that the graph nodes expect.

    Fields that aren't available yet (eval_score, drafted_sections, etc.)
    are left out — LangGraph handles missing fields gracefully, and each
    node will populate its own fields.
    """
    return {
        # Identification
        "tender_id": lead.lead_id,
        "tender_title": lead.title,
        "source_portal": lead.source_portal,
        "source_url": lead.source_url,
        "tender_raw_text": lead.description,
        "submission_deadline": lead.submission_deadline,
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "tender_document_path": None,

        # Initialise counters
        "escalation_count": 0,
        "assembly_retry_count": 0,

        # Initialise lists (required for Annotated[list, operator.add] reducers)
        "error_messages": [],
        "audit_log": [],
    }