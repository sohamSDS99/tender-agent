"""
Multi-model LLM client using Qwen API (OpenAI-compatible).

Replaces the Anthropic-based client. Uses the openai SDK pointed
at Alibaba Cloud's DashScope endpoint.

Model tiers:
  FAST     -> qwen3.5-flash   (scoring, classification)
  STANDARD -> qwen3.5-plus    (section drafting, RAG Q&A)
  ADVANCED -> qwen3-max       (compliance reasoning)
"""

from __future__ import annotations

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Maps business purpose to a specific Qwen model."""
    FAST = "qwen3.5-flash"
    STANDARD = "qwen3.5-plus"
    ADVANCED = "qwen3-max"


# Cost per 1M tokens (USD)
MODEL_COSTS = {
    "qwen3.5-flash":  {"input": 0.07,  "output": 0.26},
    "qwen3.5-plus":   {"input": 0.26,  "output": 1.56},
    "qwen3-max":      {"input": 0.78,  "output": 3.90},
}


@dataclass
class LLMUsage:
    """Token usage from a single LLM call."""
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


@dataclass
class LLMResponse:
    """Standardised response from any model tier."""
    content: str
    usage: LLMUsage
    model: str
    stop_reason: Optional[str] = None


@dataclass
class MultiModelClient:
    """
    Unified client that routes requests to the appropriate Qwen model
    based on the requested tier.

    In dry-run mode (no API key), returns deterministic mock responses
    so you can test the full pipeline without spending money.
    """

    api_key: str = ""
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    dry_run: bool = True
    _client: Optional[OpenAI] = field(default=None, init=False, repr=False)

    # Running cost accumulator
    total_cost_usd: float = field(default=0.0, init=False)
    total_input_tokens: int = field(default=0, init=False)
    total_output_tokens: int = field(default=0, init=False)
    call_count: int = field(default=0, init=False)

    def __post_init__(self):
        if self.api_key and not self.dry_run:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            logger.info(
                "Qwen client initialized",
                extra={"base_url": self.base_url},
            )
        else:
            logger.info("LLM client running in DRY-RUN mode (no API calls)")

    def complete(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.STANDARD,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        Send a prompt to the specified model tier and return a
        standardised LLMResponse.

        In dry-run mode, returns a mock response with deterministic
        token counts for testing.
        """
        model = tier.value
        start_time = time.time()

        if self.dry_run:
            return self._mock_response(prompt, model, start_time)

        if self._client is None:
            raise RuntimeError(
                "LLM client not initialized. Provide a valid "
                "DASHSCOPE_API_KEY or set DRY_RUN=true."
            )

        # Build messages list (system prompt goes as first message)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call Qwen via OpenAI-compatible API
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(
                "Qwen API call failed",
                extra={"model": model, "error": str(e)},
            )
            raise

        # Parse response (OpenAI format)
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        latency_ms = (time.time() - start_time) * 1000

        # Accumulate totals
        self.total_cost_usd += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

        llm_usage = LLMUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

        logger.info(
            "LLM call complete",
            extra={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": f"${cost:.6f}",
                "latency_ms": f"{latency_ms:.0f}",
            },
        )

        return LLMResponse(
            content=content,
            usage=llm_usage,
            model=model,
            stop_reason=choice.finish_reason,
        )

    def _mock_response(
        self, prompt: str, model: str, start_time: float
    ) -> LLMResponse:
        """
        Return a deterministic mock response for dry-run testing.
        Mimics realistic token counts and cost calculations.
        """
        input_tokens = max(len(prompt) // 4, 10)
        output_tokens = 150

        mock_content = (
            f"[DRY-RUN: {model}] This is a mock response. "
            f"In production, {model} would process your "
            f"{input_tokens}-token prompt and generate a real response. "
            f"The tender agent is working correctly in test mode."
        )

        cost = self._calculate_cost(model, input_tokens, output_tokens)
        latency_ms = (time.time() - start_time) * 1000

        self.total_cost_usd += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

        return LLMResponse(
            content=mock_content,
            usage=LLMUsage(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
            ),
            model=model,
            stop_reason="stop",
        )

    @staticmethod
    def _calculate_cost(
        model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate USD cost based on Qwen per-million-token pricing."""
        costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
        return (
            (input_tokens / 1_000_000) * costs["input"]
            + (output_tokens / 1_000_000) * costs["output"]
        )

    def get_cost_summary(self) -> dict:
        """Return accumulated cost and usage statistics."""
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "call_count": self.call_count,
        }


# Singleton instance — import this everywhere:
# from src.utils.llm_client import llm_client, ModelTier
llm_client = MultiModelClient()
