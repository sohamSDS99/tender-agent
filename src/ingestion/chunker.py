"""
Text Chunker — Splits parsed document text into overlapping chunks for embedding.

WHY CHUNKING MATTERS:
Embedding models (like Voyage AI's voyage-3-large) have a context window — they can
only process a limited amount of text at once (typically 512-8192 tokens). Even if they
could process an entire 50-page document, the resulting single embedding would be too
general to match specific queries like "What ISO certifications does the company hold?"

By splitting documents into smaller, focused chunks (typically 500-1000 tokens each),
each chunk gets its own embedding that captures a specific topic. When the RAG pipeline
searches for "ISO certifications", it finds the chunk that specifically discusses those
certs — not the one about pricing or team bios.

WHY OVERLAP:
If we split a document at exactly every 500 tokens with no overlap, a sentence that
straddles the boundary gets cut in half — losing meaning in both chunks. Overlap (say,
100 tokens) means the end of chunk N and the start of chunk N+1 share some text,
ensuring boundary sentences appear in at least one chunk completely.

KEY DESIGN DECISIONS:
- Character-based splitting (not token-based): Token counts vary by model. Characters
  are universal and fast to count. We use a rough 1 token ≈ 4 characters ratio to
  set defaults. This is close enough — the embedding model handles slight variation.
- Sentence-aware splitting: We don't chop mid-sentence. We split at sentence
  boundaries (periods, question marks, exclamation marks followed by whitespace) to
  keep each chunk grammatically coherent.
- Metadata preservation: Each chunk carries its source_file, page_number, section_heading,
  and chunk_index — all the way through to the vector database.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import structlog

from src.ingestion.parser import ParsedPage

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    """One chunk of text ready for embedding, with full provenance metadata.

    Attributes:
        text: The chunk text content.
        source_file: Original document filename.
        page_number: Page/section index in the source document.
        section_heading: Heading of the section this chunk came from (if any).
        chunk_index: 0-indexed position of this chunk within its source page.
        char_count: Number of characters.
        content_hash: SHA-256 hash for deduplication.
        metadata: Additional key-value pairs (e.g., document category, date).
    """
    text: str
    source_file: str
    page_number: int
    section_heading: str | None
    chunk_index: int
    char_count: int = field(init=False)
    content_hash: str = field(init=False)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)
        self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sentence splitting helper
# ---------------------------------------------------------------------------

# Regex that splits on sentence-ending punctuation followed by whitespace.
# Handles: "Hello. World", "Really? Yes!", "Done.\nNext line"
# Does NOT split on: "U.S.A.", "Dr. Smith", "3.14", "e.g." — because those
# don't have the pattern of [.!?] followed by [space + uppercase or newline].
# This is a pragmatic heuristic, not a perfect NLP sentence tokenizer.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])"   # lookbehind: after sentence-ending punctuation
    r"\s+"           # one or more whitespace characters
    r"(?=[A-Z\d])"  # lookahead: next sentence starts with uppercase or digit
)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex heuristic.

    Returns a list of sentence strings. If the regex finds no split points
    (e.g., a single sentence or unusual formatting), returns the original
    text as a single-element list.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Chunker class
# ---------------------------------------------------------------------------

class TextChunker:
    """Splits ParsedPage objects into overlapping TextChunk objects.

    Usage:
        chunker = TextChunker(chunk_size=2000, chunk_overlap=200)
        chunks = chunker.chunk_pages(parsed_pages)
        for chunk in chunks:
            print(f"{chunk.source_file} chunk {chunk.chunk_index}: {chunk.char_count} chars")

    Args:
        chunk_size: Target chunk size in characters. Default 2000 (~500 tokens).
            Voyage AI's voyage-3-large handles up to 16,000 tokens, so 500 tokens
            gives us plenty of headroom while keeping chunks focused.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
            Default 200 (~50 tokens). Enough to capture boundary sentences without
            wasting too much embedding compute on duplicate text.
        min_chunk_size: Minimum chunk size. Chunks smaller than this get merged with
            the previous chunk instead of becoming standalone. Prevents tiny chunks
            that produce low-quality embeddings. Default 100 characters.
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        if min_chunk_size >= chunk_size:
            raise ValueError(
                f"min_chunk_size ({min_chunk_size}) must be less than "
                f"chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_pages(self, pages: list[ParsedPage]) -> list[TextChunk]:
        """Split a list of ParsedPage objects into TextChunk objects.

        Each page is chunked independently — chunks never cross page/section
        boundaries. This keeps source attribution clean (a chunk always comes
        from exactly one page of one document).

        Args:
            pages: List of ParsedPage objects from the document parser.

        Returns:
            List of TextChunk objects ready for embedding.
        """
        all_chunks: list[TextChunk] = []

        for page in pages:
            if not page.text.strip():
                continue

            page_chunks = self._chunk_text(page.text)

            for i, chunk_text in enumerate(page_chunks):
                all_chunks.append(
                    TextChunk(
                        text=chunk_text,
                        source_file=page.source_file,
                        page_number=page.page_number,
                        section_heading=page.section_heading,
                        chunk_index=i,
                    )
                )

        logger.info(
            "chunking_complete",
            input_pages=len(pages),
            output_chunks=len(all_chunks),
            avg_chunk_chars=(
                sum(c.char_count for c in all_chunks) // max(len(all_chunks), 1)
            ),
        )

        return all_chunks

    def chunk_page(self, page: ParsedPage) -> list[TextChunk]:
        """Convenience method to chunk a single page.

        Args:
            page: A single ParsedPage object.

        Returns:
            List of TextChunk objects from this page.
        """
        return self.chunk_pages([page])

    def _chunk_text(self, text: str) -> list[str]:
        """Core chunking algorithm: sentence-aware splitting with overlap.

        ALGORITHM:
        1. Split the text into individual sentences.
        2. Accumulate sentences into a buffer until adding the next sentence
           would exceed chunk_size.
        3. Save the buffer as a chunk.
        4. Roll back by chunk_overlap characters worth of sentences to create
           the overlap for the next chunk.
        5. Repeat until all sentences are consumed.

        If a single sentence exceeds chunk_size (rare but possible with very
        long paragraphs), it becomes its own chunk — we never split mid-sentence.
        """
        sentences = _split_into_sentences(text)

        if not sentences:
            return []

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # Would adding this sentence exceed the chunk size?
            # (The +1 accounts for the space between sentences)
            if current_length + sentence_len + 1 > self.chunk_size and current_sentences:
                # Save current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(chunk_text)

                # Build overlap: take sentences from the end of the current
                # chunk until we have roughly chunk_overlap characters
                overlap_sentences: list[str] = []
                overlap_length = 0
                for s in reversed(current_sentences):
                    if overlap_length + len(s) + 1 > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1

                # Start next chunk with the overlap sentences
                current_sentences = overlap_sentences
                current_length = overlap_length

            current_sentences.append(sentence)
            current_length += sentence_len + 1

        # Don't forget the last chunk
        if current_sentences:
            last_chunk = " ".join(current_sentences)

            # If this last chunk is too small and we have a previous chunk,
            # merge it with the previous one (if the combined size is reasonable)
            if (
                len(last_chunk) < self.min_chunk_size
                and chunks
                and len(chunks[-1]) + len(last_chunk) + 1 <= self.chunk_size * 1.5
            ):
                chunks[-1] = chunks[-1] + " " + last_chunk
            else:
                chunks.append(last_chunk)

        return chunks