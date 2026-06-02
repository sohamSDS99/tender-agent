"""Submission-channel classifier.

Called from the bridge's ``handle_classify_submission_task`` the moment a
new ``TenderPursuit`` row lands (alongside ``fetch_pursuit_attachments``).

Goal: decide HOW the procurement office wants the bid delivered, so the
pursuit side panel knows which surface to render:

    form    — tender provides a downloadable fillable form (today's path)
    email   — tender description names a submission address; agent drafts
              a proposal email for HITL review
    portal  — no form, no email; operator submits via portal UI
    hybrid  — both a form AND a cover email are required
    unknown — couldn't decide; operator picks manually in the side panel

Inputs are intentionally cheap — title + description + source page HTML
(optional) + a boolean telling us whether the attachment fetcher already
found a fillable form.  All extraction is regex-based + multilingual; no
LLM call.  The classifier is pure (no I/O when ``source_page_html`` is
passed in) so the bridge can unit-test it offline.

Returned ``SubmissionClassification`` is what we POST back to AMS via
``/api/tender-pursuits/{id}/classification``.

This module is intentionally side-effect free: the bridge handler is
responsible for HTTP fetches, audit logging, and database writes.  Keep
it that way so the rules below stay testable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Channel = Literal["form", "email", "portal", "hybrid", "unknown"]


@dataclass
class SubmissionClassification:
    """The classifier's verdict for a single tender.

    ``confidence`` is on [0.0, 1.0].  Anything below 0.4 means the rules
    were ambiguous — the UI surfaces a "low confidence, please confirm"
    banner so the operator can override.
    """

    channel: Channel
    confidence: float
    contact_email: str | None = None
    contact_cc: list[str] = field(default_factory=list)
    instructions: str | None = None
    language: str | None = None
    # Human-readable explanation of which rules fired and why we picked
    # this channel.  Surfaced in /audit + the side panel's "Classifier
    # notes" disclosure so the operator can sanity-check the verdict.
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Patterns — compiled once at import time so the hot path is regex-only
# ---------------------------------------------------------------------------

# RFC-5321-ish email address.  Deliberately tolerant — procurement pages
# occasionally have trailing punctuation we'll strip post-match.
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
)

# Phrases that mean "send your bid HERE" — across the languages we see in
# the discovery feeds.  Anchored as case-insensitive substrings; the
# classifier checks whether one of these appears in an 80-char window
# around an email address before promoting it to ``contact_email``.
_SUBMIT_BY_EMAIL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # English
        r"submit (your )?(bid|proposal|offer|tender|application|response|quote)s? (by email )?to",
        r"send (your )?(bid|proposal|offer|tender|application|response|quote)s? (by email )?to",
        r"email (your )?(bid|proposal|offer|tender|application|response|quote)s? to",
        r"applications (?:are to be|should be|must be) (?:sent|emailed|submitted) to",
        r"bids? (?:are to be|should be|must be) (?:sent|emailed|submitted) (?:by email )?to",
        r"proposals? (?:are to be|should be|must be) (?:sent|emailed|submitted) (?:by email )?to",
        r"submissions? (?:are to be|should be|must be) (?:sent|emailed|submitted) (?:by email )?to",
        r"to submit (?:your )?(?:bid|proposal|offer|tender|application|response), (?:please )?email",
        r"submission (?:address|email|inbox)\s*:",
        # French
        r"envoyer (?:votre|les?) (?:offre|proposition|candidature|soumission)s? (?:par (?:e-?mail|courriel) )?(?:à|au)",
        r"adresse(?:r| de soumission)",
        r"soumettre (?:votre|les?) (?:offre|proposition)s? (?:par (?:e-?mail|courriel) )?(?:à|au)",
        # Spanish
        r"enviar (?:su|las?) (?:oferta|propuesta|solicitud|postulaci[oó]n)s? (?:por (?:correo|email) )?(?:a|al)",
        r"remitir (?:su|las?) (?:oferta|propuesta)s? (?:por (?:correo|email) )?(?:a|al)",
        r"direcci[oó]n (?:de )?(?:env[ií]o|presentaci[oó]n|recepci[oó]n)",
        # Portuguese
        r"enviar (?:sua|as?) (?:proposta|oferta|candidatura)s? (?:por (?:e-?mail|correio) )?para",
        r"submeter (?:sua|as?) (?:proposta|oferta)s? (?:por (?:e-?mail|correio) )?para",
        # German
        r"(?:senden|schicken) sie (?:ihre? )?(?:angebote?|bewerbung|vorschlag) (?:per (?:e-?mail|mail) )?an",
        r"einreichung (?:per (?:e-?mail|mail) )?an",
        # Italian
        r"inviare (?:la |le )?(?:offerta|proposta|candidatura)e? (?:via (?:e-?mail|posta) )?a",
    )
)

# Phrases around an email address that mean "this is the CONTACT for
# questions, not the SUBMISSION address".  When one of these fires
# within the same 80-char window, we down-weight the email as a
# submission target.
_CONTACT_ONLY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"for (?:questions|enquir(?:ies|y)|clarifications?|information|further (?:info|details))",
        r"contact (?:point|person|us|details?)",
        r"if you have any questions",
        r"pour (?:toute )?(?:question|information|demande de renseignements?)",
        r"para (?:cualquier |m[aá]s )?(?:consulta|informaci[oó]n|aclaraci[oó]n)",
        r"para (?:qualquer |mais )?(?:d[uú]vida|informa[cç][aã]o|esclarecimento)",
        r"bei (?:fragen|r[uü]ckfragen)",
        r"per (?:eventuali )?(?:domande|chiarimenti|informazioni)",
    )
)

# Mailboxes that almost certainly aren't a submission target.  Catches
# the "info@" / "noreply@" / "help-desk@" failure mode that the
# proximity check sometimes misses (e.g. footer-only mentions).
_GENERIC_INBOX_LOCAL_PARTS: frozenset[str] = frozenset({
    "info",
    "noreply",
    "no-reply",
    "donotreply",
    "support",
    "help",
    "helpdesk",
    "service",
    "kontakt",
    "contact",
    "contato",
    "contacto",
    "webmaster",
    "admin",
    "office",
    "press",
    "media",
    "privacy",
    "legal",
    "sales",
    "marketing",
    "feedback",
})

# Phrases that mean "you must use our portal — no email, no upload".
# We allow up to two filler words between the article and the portal noun
# ("through the SAM.gov portal", "via our online procurement system").
_PORTAL_ONLY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"submit (?:your )?(?:bid|proposal|tender|response|application|offer)s? (?:through|via|in|on|to) (?:the |our )?(?:[\w.\-]+ ){0,2}(?:portal|platform|system|website|e-?procurement)",
        r"submissions? (?:are |is )?only (?:accepted|received|allowed) (?:through|via) (?:the |our )?(?:[\w.\-]+ ){0,2}(?:portal|platform|system)",
        r"register (?:as a supplier )?(?:on|in|with) (?:the |our )?(?:[\w.\-]+ ){0,2}(?:portal|platform|system|website) (?:to )?(?:submit|bid|tender|apply)",
        r"upload (?:your )?(?:bid|proposal|tender|response|application|offer)s? (?:to|via|through|on) (?:the |our )?(?:[\w.\-]+ ){0,2}(?:portal|platform|system|website)",
        r"all documents?(?: must)?(?: be)? uploaded (?:electronically )?(?:through|via|to) (?:the |our )?(?:[\w.\-]+ ){0,2}(?:portal|platform|system|website)",
        r"electronic (?:submission|tender|bid|response)s? only",
        r"all submissions (?:must )?(?:be )?(?:submitted )?electronically (?:through|via)",
        r"submit (?:your )?(?:bid|proposal|tender|response) (?:electronically|online)",
        # FR / ES / DE / PT light coverage
        r"soumettre (?:votre|les?) (?:offre|proposition) (?:via|sur) (?:la |le |notre )?(?:[\w.\-]+ ){0,2}(?:plateforme|portail|syst[èe]me)",
        r"(?:presentar|enviar) (?:su |la )?(?:oferta|propuesta) (?:a trav[eé]s|mediante|en) (?:la |el |nuestra )?(?:[\w.\-]+ ){0,2}(?:plataforma|portal|sistema)",
        r"angebot(?:e|sabgabe) (?:über|durch) (?:die |unsere )?(?:[\w.\-]+ ){0,2}(?:plattform|portal|system)",
    )
)

# Phrases that mean "the form is attached / fill the attached form".
# We give the form channel a confidence boost when the page text
# explicitly references an attached response document — handles the
# case where the attachment fetcher *might* have missed a file.
_REFERS_TO_FORM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(?:complete|fill (?:in|out)|use) (?:the )?(?:attached|response|tender|application) form",
        r"download (?:the )?(?:response|tender|application) (?:form|template)",
        r"see (?:the )?attached (?:response|tender|application) form",
        r"the response (?:form|template) (?:is )?(?:attached|provided)",
        r"formulaire (?:de r[eé]ponse|de candidature) (?:ci-)?(?:joint|attach[eé])",
        r"formulario (?:de respuesta|de solicitud) (?:adjunto|anexo)",
    )
)

# Very lightweight language sniff — checks the first ~2 KB of the
# corpus for marker stop-words.  This is good enough for the LLM-draft
# language decision; we don't need full langdetect for v1.
_LANG_TOKENS: dict[str, tuple[str, ...]] = {
    "fr": (" le ", " la ", " les ", " et ", " est ", " pour ", " avec ", " offre ", " soumission "),
    "es": (" el ", " la ", " los ", " las ", " y ", " es ", " para ", " con ", " oferta ", " propuesta "),
    "pt": (" o ", " a ", " os ", " as ", " e ", " é ", " para ", " com ", " proposta ", " oferta "),
    "de": (" der ", " die ", " das ", " und ", " ist ", " für ", " mit ", " angebot "),
    "it": (" il ", " la ", " e ", " è ", " per ", " con ", " offerta ", " proposta "),
    # English is the implicit default — we only flag a non-en language
    # when its token count outranks English's
    "en": (" the ", " of ", " and ", " is ", " for ", " with ", " proposal ", " tender "),
}


# Header phrases that introduce a "required documents" / "what to
# include" checklist.  When matched we grab the next ~800 chars as
# the instructions narrative.
_INSTRUCTIONS_HEADERS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(?:required|requested|necessary|mandatory) documents?",
        r"documents? (?:to (?:be )?(?:included|submitted|provided)|required)",
        r"(?:submission|application|proposal) (?:must|should) (?:include|contain)",
        r"please (?:include|attach|submit)",
        r"(?:include|attach) the following",
        r"the (?:bid|proposal|response) (?:must|should) (?:include|contain)",
        r"documents? requis",
        r"documentos? (?:requeridos|necesarios|adjuntos)",
        r"documentos? (?:obrigat[oó]rios|necess[aá]rios)",
        r"erforderliche unterlagen",
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_email_trailing_punct(addr: str) -> str:
    """Some pages render an email followed by a dot or comma — strip
    common terminators so we don't store ``proc@agency.gov.``."""

    while addr and addr[-1] in ".,;:)>]}\"'":
        addr = addr[:-1]
    return addr


def _normalise_text(s: str | None) -> str:
    if not s:
        return ""
    # Collapse whitespace so window-based pattern matching works the
    # same whether the page used <br> or newlines or padded spaces.
    return re.sub(r"\s+", " ", s).strip()


def _extract_text_from_html(html: str) -> str:
    """Strip tags but preserve mailto: hrefs as inline text.

    Many procurement portals only mention the submission address inside
    ``<a href="mailto:…">``, with the visible text being something like
    "click here".  We pull the href value back into the visible string
    so the email regex still finds it.
    """

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:
        logger.warning("[classifier] BeautifulSoup parse failed: %s", exc)
        return _normalise_text(html)

    # Inline mailto: addresses into the visible text
    for a in soup.find_all("a", href=True):
        href = a.get("href") or ""
        if href.lower().startswith("mailto:"):
            mail = href.split(":", 1)[1].split("?", 1)[0]
            if mail:
                # Append once so it shows up in the text dump even if the
                # anchor's visible label was an icon / "click here".
                a.append(f" {mail}")
    return _normalise_text(soup.get_text(separator=" "))


def _window(text: str, idx: int, before: int = 100, after: int = 60) -> str:
    """Return a substring centred on idx, used for proximity scoring."""

    start = max(0, idx - before)
    end = min(len(text), idx + after)
    return text[start:end]


def _detect_language(text: str) -> str | None:
    """Cheap stop-word frequency sniff.  Returns None when English is the
    most likely language (i.e. nothing special to flag for the LLM)."""

    if not text:
        return None
    sample = (" " + text[:4000].lower() + " ")
    counts: dict[str, int] = {}
    for lang, tokens in _LANG_TOKENS.items():
        counts[lang] = sum(sample.count(tok) for tok in tokens)
    best_lang, best_count = max(counts.items(), key=lambda kv: kv[1])
    en_count = counts.get("en", 0)
    if best_count == 0:
        return None
    if best_lang == "en":
        return None
    # When English itself scored well, we want the non-English candidate
    # to win convincingly so a single "le" in an English description
    # doesn't flip the language to FR.  Threshold scales with English's
    # own count: a strong English text needs +3 to be dethroned; a
    # blank-English text only needs the non-English to clear 2.
    margin = 3 if en_count >= 3 else 0
    if best_count > en_count + margin and best_count >= 2:
        return best_lang
    return None


def _extract_instructions(text: str) -> str | None:
    """Pull the next ~800 chars after a "required documents" heading."""

    if not text:
        return None
    for pat in _INSTRUCTIONS_HEADERS:
        m = pat.search(text)
        if m:
            tail = text[m.end() : m.end() + 800].strip()
            if len(tail) >= 30:
                return tail
    return None


def _looks_like_generic_inbox(email: str) -> bool:
    local = email.split("@", 1)[0].lower()
    # Strip common variants ("info-procurement" → "info")
    bare = re.split(r"[-_.+]", local, maxsplit=1)[0]
    return bare in _GENERIC_INBOX_LOCAL_PARTS


# ---------------------------------------------------------------------------
# Email candidate scoring
# ---------------------------------------------------------------------------

@dataclass
class _EmailCandidate:
    address: str
    submit_signal: bool        # True if a submit-by-email pattern fires nearby
    contact_only_signal: bool  # True if a "for questions" pattern fires nearby
    generic_inbox: bool        # True if local-part matches the generic list
    score: float = 0.0
    window_text: str = ""


def _score_email_candidates(text: str) -> list[_EmailCandidate]:
    """Walk every email mention and score each by proximity to
    submission vs contact-only phrases.

    Higher score = more likely to be the submission address.
    """

    seen: dict[str, _EmailCandidate] = {}
    for m in _EMAIL_RE.finditer(text):
        addr = _strip_email_trailing_punct(m.group(0))
        if not addr:
            continue
        # Skip likely-noise: image placeholders, schema-mailto tokens
        if addr.lower().startswith(("noreply@", "no-reply@", "donotreply@")):
            # We still record these so we can debug, but with a heavy
            # penalty — they are almost never the real submission address.
            pass
        window = _window(text, m.start())
        submit = any(p.search(window) for p in _SUBMIT_BY_EMAIL_PATTERNS)
        contact = any(p.search(window) for p in _CONTACT_ONLY_PATTERNS)
        generic = _looks_like_generic_inbox(addr)

        score = 0.0
        if submit:
            score += 1.0
        if contact:
            score -= 0.8
        if generic:
            score -= 0.4
        # Prefer the first mention as a tie-break (procurement sections
        # tend to be at the top of the page).
        score -= 0.001 * m.start()

        existing = seen.get(addr.lower())
        if existing is None or score > existing.score:
            seen[addr.lower()] = _EmailCandidate(
                address=addr,
                submit_signal=submit,
                contact_only_signal=contact,
                generic_inbox=generic,
                score=score,
                window_text=window.strip(),
            )

    return sorted(seen.values(), key=lambda c: c.score, reverse=True)


# ---------------------------------------------------------------------------
# Optional source-page fetch — pure helper, only used when the bridge
# wants the classifier to do the GET itself instead of passing HTML in.
# ---------------------------------------------------------------------------

_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_source_page(source_url: str, *, timeout: float = 15.0) -> str:
    """GET the tender's source page and return its visible text.

    Returns an empty string on any HTTP / parsing error — the caller
    treats "no page text" as "rely on description alone", which is a
    safe degradation path.
    """

    try:
        resp = httpx.get(
            source_url,
            headers=_BROWSER_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").lower()
        if "html" not in content_type and "xml" not in content_type:
            return ""
        return _extract_text_from_html(resp.text)
    except Exception as exc:
        logger.info("[classifier] source page fetch failed (%s): %s", source_url, exc)
        return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def classify_submission(
    *,
    tender_title: str,
    tender_description: str,
    source_url: str | None = None,
    source_page_text: str | None = None,
    has_form_attachment: bool = False,
    fetch_source_when_missing: bool = True,
) -> SubmissionClassification:
    """Classify the submission channel for a single tender.

    Args:
        tender_title: Title from the discovery feed.
        tender_description: Description / summary from the discovery feed.
        source_url: Canonical tender URL (used only when we need to GET the
            source page ourselves).
        source_page_text: Pre-extracted visible text from the source page,
            if the caller already has it.  Saves a HTTP round-trip.
        has_form_attachment: True when the attachment fetcher returned at
            least one fillable form for this pursuit.
        fetch_source_when_missing: When ``source_page_text`` is empty and
            ``source_url`` is set, fetch the page ourselves.  Disabled in
            tests so the function stays pure.

    Returns:
        ``SubmissionClassification`` describing the channel, an extracted
        recipient when applicable, and any instructions narrative.
    """

    # ------------------------------------------------------------------
    # 1) Build the text corpus the rules run against
    # ------------------------------------------------------------------
    page_text = _normalise_text(source_page_text)
    if not page_text and source_url and fetch_source_when_missing:
        page_text = fetch_source_page(source_url)

    corpus_parts: list[str] = [
        _normalise_text(tender_title),
        _normalise_text(tender_description),
        page_text,
    ]
    corpus = " \n ".join(p for p in corpus_parts if p)

    if not corpus:
        return SubmissionClassification(
            channel="unknown",
            confidence=0.0,
            reasoning="No corpus to classify against (empty title/description/page).",
        )

    reasoning: list[str] = []

    # ------------------------------------------------------------------
    # 2) Pull signals
    # ------------------------------------------------------------------
    email_candidates = _score_email_candidates(corpus)
    has_strong_email_submit = any(c.submit_signal for c in email_candidates) and any(
        c.score > 0 for c in email_candidates
    )
    has_portal_phrase = any(p.search(corpus) for p in _PORTAL_ONLY_PATTERNS)
    has_form_reference = any(p.search(corpus) for p in _REFERS_TO_FORM_PATTERNS)

    if email_candidates:
        reasoning.append(
            f"email candidates: "
            + ", ".join(
                f"{c.address}(submit={c.submit_signal},contact={c.contact_only_signal},gen={c.generic_inbox})"
                for c in email_candidates[:3]
            )
        )
    if has_portal_phrase:
        reasoning.append("portal-only phrase matched")
    if has_form_reference:
        reasoning.append("page refers to attached form")
    if has_form_attachment:
        reasoning.append("attachment fetcher returned a form file")

    # ------------------------------------------------------------------
    # 3) Decide the channel
    # ------------------------------------------------------------------
    primary_email: _EmailCandidate | None = (
        email_candidates[0] if email_candidates and email_candidates[0].score > 0 else None
    )

    # We treat the email candidate as "real" only if it has a positive
    # score (submit signal present, or at least no contact-only signal
    # and not a generic inbox).
    has_real_email = primary_email is not None

    if has_form_attachment and has_real_email and primary_email is not None:
        channel: Channel = "hybrid"
        confidence = 0.85
    elif has_form_attachment:
        channel = "form"
        confidence = 0.9
        primary_email = None
    elif has_strong_email_submit and has_real_email:
        channel = "email"
        confidence = 0.85
    elif has_real_email and not has_portal_phrase:
        # We found an email that doesn't look like a contact-only mention
        # but no explicit submit phrase.  Probably email, lower confidence.
        channel = "email"
        confidence = 0.6
    elif has_portal_phrase:
        channel = "portal"
        confidence = 0.7
        primary_email = None
    elif has_form_reference:
        # Page says "fill the attached form" but the fetcher didn't find
        # one — likely a fetch failure, not a real channel signal.  We
        # surface as unknown so the operator can retry the fetch.
        channel = "unknown"
        confidence = 0.25
        primary_email = None
    else:
        channel = "unknown"
        confidence = 0.15
        primary_email = None

    # Slightly down-weight low-quality email candidates: if our only
    # signal was a generic inbox with no submit phrase, drop confidence.
    if channel == "email" and primary_email is not None:
        if primary_email.generic_inbox and not primary_email.submit_signal:
            confidence = min(confidence, 0.45)

    # ------------------------------------------------------------------
    # 4) CC extraction — only when we're committing to email-mode
    # ------------------------------------------------------------------
    cc_list: list[str] = []
    if channel in ("email", "hybrid") and primary_email is not None:
        primary_lower = primary_email.address.lower()
        for cand in email_candidates[1:]:
            if cand.address.lower() == primary_lower:
                continue
            # Only promote a second candidate to CC if it ALSO sits in a
            # submission context and isn't obviously a generic inbox.
            if cand.submit_signal and not cand.contact_only_signal and not cand.generic_inbox:
                cc_list.append(cand.address)
        # Cap at 3 — anything more is almost certainly noise.
        cc_list = cc_list[:3]

    return SubmissionClassification(
        channel=channel,
        confidence=round(confidence, 3),
        contact_email=primary_email.address if primary_email else None,
        contact_cc=cc_list,
        instructions=_extract_instructions(corpus),
        language=_detect_language(corpus),
        reasoning="; ".join(reasoning) or "no rules fired",
    )


__all__ = [
    "Channel",
    "SubmissionClassification",
    "classify_submission",
    "fetch_source_page",
]
