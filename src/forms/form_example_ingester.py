"""
Form Example Ingester — extract Q&A pairs from a filled past form and
push them to the AMS so future fills can learn from them.

The flow:
  1. AMS exposes documents with category=form_example via
     GET /api/agents/<name>/form-examples?pending=1
  2. Bridge polls that endpoint each loop tick.
  3. For every returned document:
       a. Fetch the file from MinIO (via the existing helper).
       b. Run FormParser to get (field name, current value) pairs —
          for filled past forms, current_value is the answer.
       c. Embed each question with Voyage AI (voyage-3-large, 1024-dim).
       d. POST the array of {questionText, answerText, questionEmbedding,
          metadata} to the AMS ingest endpoint.
  4. AMS bulk-inserts FormExample rows and stamps the document as
     processed so the bridge stops re-picking it.

Idempotent by design: the AMS endpoint wipes prior rows for the same
documentId before inserting, so re-running ingestion gives a clean
replacement.

This module is bridge-only — no AMS-side logic lives here.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Callable

import httpx
import structlog

from .form_parser import FormParser, ParseResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

@dataclass
class QaPair:
    """A single question/answer pair extracted from a filled past form."""

    question_text: str
    answer_text: str
    question_embedding: list[float]  # voyage-3-large 1024-dim vector
    metadata: dict[str, Any]


@dataclass
class IngestionResult:
    """Outcome of processing one form_example document."""

    document_id: str
    pairs_extracted: int
    pairs_pushed: int
    skipped_reason: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Heuristics — decide whether a parsed FormField row looks like a real
# Q&A pair we want to learn from
# ---------------------------------------------------------------------------

# Drop tiny / noise answers — page numbers, dates alone, single characters.
# These come up a lot in PDF form parsers misreading numbered lists.
_MIN_QUESTION_CHARS = 4
_MIN_ANSWER_CHARS = 2

# A value composed only of digits, dashes, and whitespace is unlikely to
# be a meaningful answer; usually a page number or section index. We
# still keep numeric ANSWERS if the question explicitly asks for a number.
_PURE_NUMERIC = re.compile(r"^[\d\-\s./]+$")


def _is_meaningful_pair(question: str, answer: str) -> tuple[bool, str]:
    """Return (keep, reason). reason is empty when keep=True."""
    q = (question or "").strip()
    a = (answer or "").strip()

    if len(q) < _MIN_QUESTION_CHARS:
        return False, "question too short"
    if len(a) < _MIN_ANSWER_CHARS:
        return False, "answer too short / empty"
    if _PURE_NUMERIC.match(a) and not any(
        token in q.lower()
        for token in ("number", "amount", "year", "count", "qty", "$", "%", "no.")
    ):
        return False, "answer looks like a page number / index"
    return True, ""


# ---------------------------------------------------------------------------
# Voyage AI embeddings — we reuse the same model the KB uses
# ---------------------------------------------------------------------------

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"


def embed_questions_voyage(
    questions: list[str],
    *,
    api_key: str,
    timeout: float = 30.0,
    batch_size: int = 16,
) -> list[list[float]]:
    """Embed a list of question strings using Voyage AI voyage-3-large.

    Returns one 1024-dim vector per input, in the same order.
    Empty list on any failure — caller decides what to do.
    """
    if not questions:
        return []
    if not api_key:
        logger.warning("voyage_no_api_key", msg="VOYAGE_API_KEY not set; skipping embeddings")
        return []

    out: list[list[float]] = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Voyage allows up to ~128 inputs per call, but smaller batches make
    # partial-failure recovery cheaper and keep per-call latency bounded.
    try:
        with httpx.Client(timeout=timeout) as client:
            for start in range(0, len(questions), batch_size):
                chunk = questions[start : start + batch_size]
                resp = client.post(
                    _VOYAGE_URL,
                    headers=headers,
                    json={
                        "model": "voyage-3-large",
                        "input": chunk,
                        "input_type": "document",
                        "output_dimension": 1024,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                vectors = [row["embedding"] for row in data.get("data", [])]
                if len(vectors) != len(chunk):
                    logger.warning(
                        "voyage_batch_size_mismatch",
                        expected=len(chunk),
                        got=len(vectors),
                    )
                out.extend(vectors)
    except Exception as exc:
        logger.error("voyage_embed_failed", error=str(exc))
        return []

    return out


# ---------------------------------------------------------------------------
# Q&A extraction from a filled past form
# ---------------------------------------------------------------------------

def extract_qa_pairs_from_form(
    file_path: str,
    *,
    voyage_api_key: str,
    parser: FormParser | None = None,
) -> list[QaPair]:
    """Parse a filled past form into Q&A pairs ready for the AMS.

    Returns an empty list if the file can't be parsed or has no
    fields with values.
    """
    parser = parser or FormParser()
    try:
        parsed: ParseResult = parser.parse(file_path)
    except Exception as exc:
        logger.error("form_example_parse_failed", file_path=file_path, error=str(exc))
        return []

    # Keep only fields that have an answer (current_value populated).
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for field in parsed.fields:
        question = field.name
        answer = field.current_value or ""
        keep, reason = _is_meaningful_pair(question, answer)
        if not keep:
            logger.debug(
                "form_example_pair_dropped",
                question=question[:80],
                reason=reason,
            )
            continue

        meta: dict[str, Any] = {
            "fieldType": field.field_type,
            "required": field.required,
            "section": field.page_or_section or "",
        }
        # Don't store free-form metadata blobs — they tend to be huge.
        if field.options:
            meta["options"] = field.options[:20]
        candidates.append((question.strip(), answer.strip(), meta))

    if not candidates:
        logger.info(
            "form_example_no_pairs",
            file_path=file_path,
            field_count=len(parsed.fields),
        )
        return []

    # Embed all questions in one go
    embeddings = embed_questions_voyage(
        [c[0] for c in candidates],
        api_key=voyage_api_key,
    )
    if len(embeddings) != len(candidates):
        # Partial embed failure → drop everything; caller can retry
        logger.warning(
            "form_example_embed_mismatch",
            file_path=file_path,
            pairs=len(candidates),
            embeddings=len(embeddings),
        )
        return []

    return [
        QaPair(
            question_text=question,
            answer_text=answer,
            question_embedding=emb,
            metadata=meta,
        )
        for (question, answer, meta), emb in zip(candidates, embeddings)
    ]


# ---------------------------------------------------------------------------
# Bridge entry point — list pending examples, process each, post results
# ---------------------------------------------------------------------------

def process_pending_form_examples(
    *,
    ams_url: str,
    agent_name: str,
    minio_download_fn: Callable[[str, str], str],
    voyage_api_key: str,
    auth_headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    max_docs: int = 5,
) -> list[IngestionResult]:
    """Drive end-to-end ingestion for any pending form_example documents.

    Args:
        ams_url: Base AMS URL (e.g. http://localhost:3000).
        agent_name: The agent the bridge is registered as ("tender-agent").
        minio_download_fn: Callable(bucket, key) -> local file path. The
            bridge already has one; we accept it as a dependency so we
            don't pull MinIO config into this module.
        voyage_api_key: Voyage AI key for question embeddings.
        auth_headers: Bearer-auth headers if NEXUS_AGENT_API_KEY is set
            on both sides.
        timeout: Per-HTTP-call timeout in seconds.
        max_docs: Cap to keep one bridge tick bounded.

    Returns:
        A list of IngestionResult, one per document processed.
    """
    auth_headers = auth_headers or {}
    list_url = f"{ams_url}/api/agents/{agent_name}/form-examples?pending=1"
    post_url = f"{ams_url}/api/agents/{agent_name}/form-examples"

    out: list[IngestionResult] = []

    # 1) Fetch pending documents
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(list_url, headers=auth_headers)
            resp.raise_for_status()
            docs = resp.json().get("documents", []) or []
    except Exception as exc:
        logger.error("form_example_list_failed", error=str(exc))
        return out

    if not docs:
        return out

    logger.info("form_example_processing_start", pending=len(docs))

    for doc in docs[:max_docs]:
        document_id = doc.get("id") or ""
        bucket = doc.get("minioBucket") or doc.get("minio_bucket") or ""
        key = doc.get("minioKey") or doc.get("minio_key") or ""
        if not (document_id and bucket and key):
            out.append(
                IngestionResult(
                    document_id=document_id or "<unknown>",
                    pairs_extracted=0,
                    pairs_pushed=0,
                    skipped_reason="missing minio reference",
                )
            )
            continue

        # 2) Download the file to a temp path
        local_path: str | None = None
        try:
            local_path = minio_download_fn(bucket, key)
        except Exception as exc:
            out.append(
                IngestionResult(
                    document_id=document_id,
                    pairs_extracted=0,
                    pairs_pushed=0,
                    error=f"minio download failed: {exc}",
                )
            )
            continue

        # 3) Parse + embed
        try:
            pairs = extract_qa_pairs_from_form(
                local_path,
                voyage_api_key=voyage_api_key,
            )
        finally:
            # Best-effort cleanup of the temp file
            try:
                if local_path and os.path.isfile(local_path):
                    os.unlink(local_path)
            except OSError:
                pass

        if not pairs:
            # We POST anyway with an empty list so AMS stamps
            # qaExtractedAt and stops re-picking this doc.  Without
            # this, a noisy unfillable form would re-poll forever.
            payload = {"documentId": document_id, "qaPairs": []}
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(post_url, headers=auth_headers, json=payload)
                    resp.raise_for_status()
            except Exception as exc:
                out.append(
                    IngestionResult(
                        document_id=document_id,
                        pairs_extracted=0,
                        pairs_pushed=0,
                        error=f"empty-stamp post failed: {exc}",
                    )
                )
                continue
            out.append(
                IngestionResult(
                    document_id=document_id,
                    pairs_extracted=0,
                    pairs_pushed=0,
                    skipped_reason="no meaningful Q&A pairs found",
                )
            )
            continue

        # 4) Push to AMS
        payload = {
            "documentId": document_id,
            "qaPairs": [
                {
                    "questionText": p.question_text,
                    "answerText": p.answer_text,
                    "questionEmbedding": p.question_embedding,
                    "metadata": p.metadata,
                }
                for p in pairs
            ],
        }
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(post_url, headers=auth_headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            out.append(
                IngestionResult(
                    document_id=document_id,
                    pairs_extracted=len(pairs),
                    pairs_pushed=int(data.get("inserted", 0)),
                )
            )
            logger.info(
                "form_example_ingested",
                document_id=document_id,
                pairs_extracted=len(pairs),
                pairs_pushed=data.get("inserted"),
            )
        except Exception as exc:
            out.append(
                IngestionResult(
                    document_id=document_id,
                    pairs_extracted=len(pairs),
                    pairs_pushed=0,
                    error=f"AMS post failed: {exc}",
                )
            )

    return out


def search_similar_examples(
    *,
    ams_url: str,
    agent_name: str,
    question: str,
    voyage_api_key: str,
    limit: int = 5,
    min_similarity: float = 0.55,
    auth_headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> list[dict[str, Any]]:
    """Find the top-N most similar past Q&A examples for a new question.

    Used by form_filler when answering a new question on a blank form.
    Returns an empty list on any failure so the filler gracefully
    degrades to KB-only mode.
    """
    if not question or not voyage_api_key:
        return []

    embeddings = embed_questions_voyage([question], api_key=voyage_api_key)
    if not embeddings:
        return []

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{ams_url}/api/agents/{agent_name}/form-examples?op=search",
                headers=auth_headers or {},
                json={
                    "embedding": embeddings[0],
                    "limit": limit,
                    "minSimilarity": min_similarity,
                },
            )
            resp.raise_for_status()
            return resp.json().get("matches", []) or []
    except Exception as exc:
        logger.debug("form_example_search_failed", error=str(exc))
        return []
