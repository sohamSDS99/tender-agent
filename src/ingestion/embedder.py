"""
Voyage AI Embedding Client — Converts text into 1024-dimensional vector embeddings.

WHY EMBEDDINGS EXIST:
Traditional search (keyword matching) fails when the query uses different words than
the document. If someone asks "What security certifications do you hold?" but your
document says "ISO 27001 compliance achieved in 2020", keyword search misses it because
none of the exact words match.

Vector embeddings solve this. An embedding model reads text and outputs a list of numbers
(a "vector") that captures the *meaning*. Texts with similar meanings get similar vectors,
even if they use completely different words. "Security certifications" and "ISO 27001
compliance" produce vectors that are close together in the 1024-dimensional space.

WHY VOYAGE AI:
The architecture specifies Voyage AI's `voyage-3-large` model (1024 dimensions). It's
one of the best embedding models available in 2026, outperforming OpenAI's text-embedding-3
on retrieval benchmarks. At ~$0.06 per million tokens, it's also cheap — our entire
knowledge base of ~100 documents costs about $0.05 to embed once.

HOW DRY-RUN MODE WORKS:
Since you don't have a Voyage AI API key yet (DRY_RUN=true), this client generates
deterministic random vectors from the text content. The vectors are seeded by the text's
hash, so the same text always produces the same fake vector — this keeps test behaviour
consistent. Once you get a real API key, flip DRY_RUN=false and re-run the embedding
pipeline to replace fake vectors with real ones.

BATCH PROCESSING:
Voyage AI accepts up to 128 texts per API call. Sending texts one at a time would be
128x slower and waste money on HTTP overhead. This client batches automatically.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Voyage AI's voyage-3-large outputs 1024-dimensional vectors
EMBEDDING_DIMENSIONS: int = 1024

# Maximum texts per API call (Voyage AI limit)
MAX_BATCH_SIZE: int = 128

# Default model name
DEFAULT_MODEL: str = "voyage-3-large"


# ---------------------------------------------------------------------------
# Result data structure
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    """Result from an embedding operation.

    Attributes:
        embeddings: List of vectors (each is a list of floats).
        model: The model that produced these embeddings.
        token_count: Total tokens processed (for cost tracking).
        is_dry_run: Whether these are fake (random) vectors.
        duration_seconds: How long the API call took.
    """
    embeddings: list[list[float]]
    model: str
    token_count: int
    is_dry_run: bool
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------

class VoyageEmbedder:
    """Client for computing vector embeddings via Voyage AI's API.

    Usage:
        # Dry-run mode (no API key needed)
        embedder = VoyageEmbedder(dry_run=True)
        result = embedder.embed_texts(["Hello world", "Safety Data Sheet"])
        print(len(result.embeddings))       # 2
        print(len(result.embeddings[0]))    # 1024

        # Real mode (requires VOYAGE_API_KEY env var)
        embedder = VoyageEmbedder(dry_run=False)
        result = embedder.embed_texts(["ISO 27001 certification"])

    Args:
        dry_run: If True, return deterministic random vectors instead of
            calling the Voyage AI API. Default reads from DRY_RUN env var.
        api_key: Voyage AI API key. If None, reads from VOYAGE_API_KEY env var.
        model: Embedding model name. Default: voyage-3-large.
    """

    def __init__(
        self,
        dry_run: bool | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        # Resolve dry_run from env if not explicitly set
        if dry_run is None:
            dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")
        self.dry_run = dry_run

        self.model = model
        self._api_key = api_key or os.getenv("VOYAGE_API_KEY", "")

        if not self.dry_run and not self._api_key:
            raise ValueError(
                "VOYAGE_API_KEY is required when DRY_RUN is not enabled. "
                "Set it in your .env file or pass api_key= to VoyageEmbedder()."
            )

        # Track cumulative token usage for cost monitoring
        self.total_tokens_used: int = 0
        self.total_api_calls: int = 0

        logger.info(
            "embedder_initialized",
            model=self.model,
            dry_run=self.dry_run,
            dimensions=EMBEDDING_DIMENSIONS,
        )

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Embed a list of texts into vectors.

        Automatically batches if the input exceeds MAX_BATCH_SIZE.

        Args:
            texts: List of text strings to embed. Can be 1 to any number.

        Returns:
            EmbeddingResult with one embedding per input text (same order).

        Raises:
            ValueError: If texts list is empty.
            RuntimeError: If the API call fails (real mode only).
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        start_time = time.time()

        if self.dry_run:
            result = self._embed_dry_run(texts)
        else:
            result = self._embed_real(texts)

        result.duration_seconds = time.time() - start_time

        self.total_tokens_used += result.token_count
        self.total_api_calls += 1

        logger.info(
            "embedding_complete",
            texts_count=len(texts),
            tokens_used=result.token_count,
            duration_s=round(result.duration_seconds, 3),
            dry_run=result.is_dry_run,
            cumulative_tokens=self.total_tokens_used,
        )

        return result

    def embed_single(self, text: str) -> list[float]:
        """Convenience method to embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A single embedding vector (list of 1024 floats).
        """
        result = self.embed_texts([text])
        return result.embeddings[0]

    def get_cost_estimate(self) -> float:
        """Estimate the total cost based on tokens used.

        Voyage AI pricing (as of 2026): ~$0.06 per 1M tokens for voyage-3-large.

        Returns:
            Estimated cost in USD.
        """
        return self.total_tokens_used * 0.06 / 1_000_000

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _embed_dry_run(self, texts: list[str]) -> EmbeddingResult:
        """Generate deterministic fake embeddings from text hashes.

        WHY DETERMINISTIC:
        Using the text's hash as a random seed means the same text always
        produces the same fake vector. This makes tests reproducible —
        you can assert that two identical texts produce identical embeddings.

        The vectors are normalised to unit length (like real embeddings)
        so cosine similarity calculations work correctly in tests.
        """
        import random

        embeddings: list[list[float]] = []
        estimated_tokens = 0

        for text in texts:
            # Seed the random generator with the text's hash for determinism
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            rng = random.Random(text_hash)

            # Generate a random vector
            raw = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIMENSIONS)]

            # Normalise to unit length (L2 norm = 1), like real embeddings
            magnitude = sum(x * x for x in raw) ** 0.5
            normalised = [x / magnitude for x in raw]

            embeddings.append(normalised)

            # Rough token estimate: 1 token ≈ 4 characters
            estimated_tokens += len(text) // 4

        return EmbeddingResult(
            embeddings=embeddings,
            model=f"{self.model}-dry-run",
            token_count=estimated_tokens,
            is_dry_run=True,
        )

    def _embed_real(self, texts: list[str]) -> EmbeddingResult:
        """Call the Voyage AI API to compute real embeddings.

        Handles batching automatically — if you pass 300 texts, this sends
        3 API calls of 128, 128, and 44 texts respectively, then combines
        the results.

        IMPORTANT: This requires the `httpx` library for HTTP calls. We use
        httpx instead of requests because it's already a dependency of the
        Anthropic SDK (installed in Step 4), so it adds no new dependencies.
        """
        import httpx

        all_embeddings: list[list[float]] = []
        total_tokens = 0

        # Process in batches of MAX_BATCH_SIZE
        for batch_start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[batch_start : batch_start + MAX_BATCH_SIZE]

            logger.debug(
                "embedding_batch",
                batch_start=batch_start,
                batch_size=len(batch),
                total_texts=len(texts),
            )

            response = httpx.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": batch,
                    "input_type": "document",
                },
                timeout=60.0,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Voyage AI API error {response.status_code}: {response.text}"
                )

            data = response.json()

            # Extract embeddings — API returns them in the same order as input
            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)

            # Track token usage
            total_tokens += data.get("usage", {}).get("total_tokens", 0)

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self.model,
            token_count=total_tokens,
            is_dry_run=False,
        )