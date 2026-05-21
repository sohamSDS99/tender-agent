"""Tender-attachment fetcher.

Called from the bridge's ``handle_fetch_attachments_task`` when the
operator selects a tender for pursuit and AMS asks us to populate the
pursuit with its actual form files.

Workflow:

    1. AMS sends ``(sourceUrl, attachmentUrls[])`` for one pursuit via
       ``GET /api/tender-pursuits/<id>/source``.
    2. If ``attachmentUrls`` is non-empty (the discovery module already
       extracted them from the API response), we download those directly.
    3. Otherwise we fetch ``sourceUrl`` as HTML and walk every ``<a
       href=...>`` whose target looks like a downloadable office file
       (PDF/DOCX/XLSX/ZIP/ODT/RTF).
    4. Each candidate is downloaded with browser-shaped headers,
       redirects followed, a 60s timeout, and a 25 MiB per-file ceiling.
    5. Each file is scored by filename for "how form-like is this?" and
       the highest score is marked ``is_primary``. Operator can
       override later through the UI.

This module does NOT touch AMS or MinIO — the bridge handler does that
after we hand back the in-memory payload. That keeps the fetcher pure
and unit-testable.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Extensions we treat as "downloadable form-like attachments".  ZIPs are
# included because many procurement portals bundle the response template
# inside one (Brazil PNCP, BOAMP France, some SAM.gov).
_DOWNLOADABLE_EXTS: tuple[str, ...] = (
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".odt",
    ".ods",
    ".rtf",
    ".zip",
)

# Filename tokens that *raise* a candidate's form-likelihood score.  Many
# real tenders title the response document with one of these words.
_FORM_TOKENS: tuple[str, ...] = (
    "form",
    "response",
    "submission",
    "proposal",
    "template",
    "questionnaire",
    "application",
    "bidder",
    "tender_response",
    "rfp_response",
    "rfi_response",
    "rfq_response",
    "offre",       # FR
    "formulaire",  # FR
    "formulario",  # ES/PT
    "respuesta",   # ES
    "proposta",    # PT
    "antrag",      # DE
    "anbot",       # NO
)

# Filename tokens that *lower* the score — these documents are normally
# supporting context (specs, drawings) rather than the response form.
_CONTEXT_TOKENS: tuple[str, ...] = (
    "specification",
    "specifications",
    "specs",
    "terms_of_reference",
    "tor",
    "appendix",
    "annex",
    "schedule",
    "drawing",
    "technical_spec",
    "cahier_des_charges",  # FR
    "pliego",              # ES
    "edital",              # PT
    "leistungsbeschreibung",  # DE
)

# Cap per-file download size.  Real tender forms are <5 MiB; anything
# bigger is probably a CAD pack or a media kit and we skip it.
_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MiB

# Don't follow more than this many candidate links per source page.
# Operator-side experience: 20+ attachments == hard to review anyway.
_MAX_ATTACHMENTS_PER_PURSUIT = 15

_REQUEST_TIMEOUT_S = 60.0

# Browser-shaped headers — many procurement CDNs return 403 to anything
# that smells like a script (CanadaBuys taught us this in Session 5).
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/pdf,application/octet-stream;q=0.8,*/*;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FetchedAttachment:
    """One downloaded file ready for MinIO upload."""

    filename: str
    local_path: str
    mime_type: str
    size_bytes: int
    source_url: str
    score: float
    is_primary: bool = False
    # Filled by the bridge after MinIO upload — kept here so the same
    # dataclass can be serialised to the AMS /attachments POST body.
    minio_key: str | None = None


@dataclass
class FetchResult:
    """Top-level return from ``fetch_pursuit_attachments``."""

    attachments: list[FetchedAttachment] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    # Why no primary was picked — surfaced to AMS so the UI can show a
    # useful "we couldn't find a form" hint instead of a generic spinner
    # forever.
    note: str | None = None

    @property
    def primary(self) -> FetchedAttachment | None:
        for a in self.attachments:
            if a.is_primary:
                return a
        return None


# ---------------------------------------------------------------------------
# Filename scoring — picks "the form" from a pile of attachments
# ---------------------------------------------------------------------------


def score_form_likelihood(filename: str) -> float:
    """Return a score in roughly [-1.0, 2.0] for how form-like a file is.

    Higher = more likely to be the actual response template the
    operator needs to fill out.  Used to auto-pick ``is_primary``.
    """

    name = filename.lower()
    stem = Path(name).stem.replace("-", "_").replace(" ", "_")
    ext = Path(name).suffix

    score = 0.0

    # Strong form indicators
    for tok in _FORM_TOKENS:
        if tok in stem:
            score += 1.0
            break  # don't double-count "tender_response_form"

    # Context-only indicators pull the score down
    for tok in _CONTEXT_TOKENS:
        if tok in stem:
            score -= 0.6
            break

    # Editable office formats outrank PDFs because they have real form
    # fields we can fill programmatically.
    if ext in (".docx", ".xlsx", ".doc", ".xls", ".odt", ".ods"):
        score += 0.4
    elif ext == ".pdf":
        score += 0.1  # may have AcroForm fields, but often flat
    elif ext == ".zip":
        score -= 0.2  # extra layer of unpacking
    elif ext == ".rtf":
        score += 0.2

    # Penalise obviously huge filenames (numbered attachments like
    # "Annex_7_drawings_set_3.pdf" — context, not the form).
    if re.search(r"annex[_-]?\d+", stem):
        score -= 0.4

    return score


def pick_primary(attachments: Sequence[FetchedAttachment]) -> int | None:
    """Index of the best primary candidate, or None if nothing qualifies."""

    if not attachments:
        return None

    best_idx = 0
    best_score = attachments[0].score
    for i, att in enumerate(attachments[1:], start=1):
        if att.score > best_score:
            best_score = att.score
            best_idx = i

    # If the top score is still negative (everything is context-only),
    # we don't fake a primary — the operator should upload manually.
    if best_score < 0.0:
        return None
    return best_idx


# ---------------------------------------------------------------------------
# HTML scraping — find attachment links when the API didn't give us any
# ---------------------------------------------------------------------------


def extract_attachment_urls_from_html(html: str, base_url: str) -> list[str]:
    """Find downloadable attachment URLs on a tender's detail page.

    Walks every ``<a href>`` and keeps those whose target ends in a
    known office-doc extension.  Resolves relative URLs against
    ``base_url``.  Deduplicates while preserving page order so the
    first PDF/DOCX you'd see scrolling the page stays first.
    """

    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue

        # Some portals embed an "onclick=window.open('...')" — we ignore
        # those (browser-only) and trust the visible href.
        target = urljoin(base_url, href)

        # Strip query string for extension match but keep it on the URL
        # we actually download (auth tokens often live there).
        path = urlparse(target).path.lower()
        if not any(path.endswith(ext) for ext in _DOWNLOADABLE_EXTS):
            continue

        if target in seen:
            continue
        seen.add(target)
        found.append(target)

        if len(found) >= _MAX_ATTACHMENTS_PER_PURSUIT:
            break

    return found


def _filename_from_response(resp: httpx.Response, fallback_url: str) -> str:
    """Best-effort original filename for a downloaded attachment."""

    # 1. Content-Disposition header (most portals set this)
    cd = resp.headers.get("content-disposition") or ""
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        # Trim any path components a malicious server might inject
        candidate = os.path.basename(candidate)
        if candidate:
            return candidate

    # 2. Last path segment of the URL
    parsed = urlparse(fallback_url)
    name = os.path.basename(parsed.path) or "attachment"
    # If the URL had no extension, infer one from Content-Type
    if "." not in name:
        ct = (resp.headers.get("content-type") or "").split(";")[0].strip()
        ext = mimetypes.guess_extension(ct) or ""
        if ext:
            name = f"{name}{ext}"
    return name


# ---------------------------------------------------------------------------
# Download orchestration
# ---------------------------------------------------------------------------


def _download_one(
    client: httpx.Client,
    url: str,
    *,
    dest_dir: Path,
) -> FetchedAttachment | None:
    """Download one URL into ``dest_dir`` and return a FetchedAttachment.

    Returns None (and logs) when the download exceeds the size cap or
    the response status isn't 2xx.
    """

    try:
        with client.stream("GET", url, headers=_BROWSER_HEADERS) as resp:
            resp.raise_for_status()

            # Reject bigger-than-cap up front via header if present.
            cl = resp.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > _MAX_FILE_BYTES:
                logger.warning(
                    "[attachment] %s skipped — declared size %s exceeds cap",
                    url, cl,
                )
                return None

            filename = _filename_from_response(resp, url)
            local_path = dest_dir / filename
            # Avoid clobber if the page links to two files with the same
            # final segment (e.g. "form.pdf" from different folders).
            if local_path.exists():
                stem, suffix = local_path.stem, local_path.suffix
                i = 1
                while local_path.exists():
                    local_path = dest_dir / f"{stem}__{i}{suffix}"
                    i += 1

            written = 0
            with open(local_path, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                    fh.write(chunk)
                    written += len(chunk)
                    if written > _MAX_FILE_BYTES:
                        fh.close()
                        local_path.unlink(missing_ok=True)
                        logger.warning(
                            "[attachment] %s aborted — exceeded %d bytes",
                            url, _MAX_FILE_BYTES,
                        )
                        return None

            mime_type = (
                resp.headers.get("content-type", "").split(";")[0].strip()
                or mimetypes.guess_type(filename)[0]
                or "application/octet-stream"
            )

            return FetchedAttachment(
                filename=filename,
                local_path=str(local_path),
                mime_type=mime_type,
                size_bytes=written,
                source_url=url,
                score=score_form_likelihood(filename),
            )

    except httpx.HTTPError as exc:
        logger.warning("[attachment] %s download failed: %s", url, exc)
        return None
    except Exception as exc:  # noqa: BLE001 — never fail the whole batch
        logger.exception("[attachment] %s unexpected failure: %s", url, exc)
        return None


def fetch_pursuit_attachments(
    *,
    source_url: str | None,
    api_attachment_urls: Iterable[str] | None = None,
    dest_dir: str | None = None,
) -> FetchResult:
    """Fetch all attachments for one TenderPursuit.

    Args:
        source_url: The DiscoveredTender's detail page URL. Used as
            fallback when the discovery module didn't extract direct
            attachment URLs from the API response.
        api_attachment_urls: URLs the discovery module already harvested
            (e.g. SAM.gov ``resourceLinks``).  When non-empty we skip
            HTML scraping entirely.
        dest_dir: Optional destination directory for the downloaded
            files. A unique temp dir is created when omitted; the
            caller is responsible for cleaning it up after MinIO upload.

    Returns:
        ``FetchResult`` with the downloaded attachments, the primary
        marker set, and a human-readable ``note`` when no primary
        could be picked.
    """

    work_dir = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="pursuit_att_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    candidate_urls: list[str] = list(api_attachment_urls or [])

    # Dedupe whatever the API gave us before any HTML round trip.
    seen: set[str] = set()
    candidate_urls = [u for u in candidate_urls if not (u in seen or seen.add(u))]

    with httpx.Client(
        timeout=_REQUEST_TIMEOUT_S,
        follow_redirects=True,
        headers=_BROWSER_HEADERS,
    ) as client:

        # Fall back to HTML scraping when the API gave us nothing.
        if not candidate_urls and source_url:
            try:
                resp = client.get(source_url)
                resp.raise_for_status()
                html = resp.text
                candidate_urls = extract_attachment_urls_from_html(
                    html, base_url=str(resp.url)
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    "[attachment] could not fetch source page %s: %s",
                    source_url, exc,
                )
                return FetchResult(
                    note=f"Source page fetch failed: {exc}",
                )

        if not candidate_urls:
            return FetchResult(
                note=(
                    "No downloadable attachments found on the tender's source page. "
                    "Operator will need to upload the form manually."
                ),
            )

        downloaded: list[FetchedAttachment] = []
        skipped: list[dict[str, str]] = []

        for url in candidate_urls[:_MAX_ATTACHMENTS_PER_PURSUIT]:
            att = _download_one(client, url, dest_dir=work_dir)
            if att is None:
                skipped.append({"url": url, "reason": "download_failed_or_too_large"})
                continue
            downloaded.append(att)

    if not downloaded:
        return FetchResult(skipped=skipped, note="All candidate attachments failed to download.")

    # Pick the primary by score
    primary_idx = pick_primary(downloaded)
    if primary_idx is not None:
        downloaded[primary_idx].is_primary = True
        note = None
    else:
        note = (
            "Attachments were found but none looked like a fillable form. "
            "Operator should choose the primary manually or upload one."
        )

    return FetchResult(attachments=downloaded, skipped=skipped, note=note)


__all__ = [
    "FetchedAttachment",
    "FetchResult",
    "extract_attachment_urls_from_html",
    "fetch_pursuit_attachments",
    "pick_primary",
    "score_form_likelihood",
]
