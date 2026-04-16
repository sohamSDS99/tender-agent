"""
Anthropic multi-model client with cost tracking and dry-run support.

This is the single entry point for all LLM calls in the agent.
Every node calls this module instead of using the Anthropic SDK directly.

Usage:
    from src.utils.llm_client import llm_client, ModelTier

    # Quick scoring with Haiku (cheapest)
    result = llm_client.call(
        prompt="Is this tender relevant to EHS software?",
        tier=ModelTier.FAST,
        system="You are a tender eligibility classifier.",
    )
    print(result.text)
    print(f"Cost: ${result.cost_usd:.6f}")

    # Drafting with Sonnet (balanced)
    result = llm_client.call(
        prompt="Draft the executive summary section.",
        tier=ModelTier.STANDARD,
        system="You are a professional bid writer.",
        max_tokens=2000,
    )

    # Complex compliance analysis with Opus (most capable)
    result = llm_client.call(
        prompt="Analyse whether our ISO 27001 cert meets this requirement.",
        tier=ModelTier.ADVANCED,
        system="You are a compliance advisor.",
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from anthropic import Anthropic

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

class ModelTier(Enum):
    """
    Model tiers map to cost/capability levels.

    Nodes specify the TIER they need, not the model name.
    This means we can swap models without changing node code.
    """

    FAST = "fast"           # Haiku — scoring, classification
    STANDARD = "standard"   # Sonnet — drafting, RAG Q&A
    ADVANCED = "advanced"   # Opus — compliance reasoning


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str               # API model string
    display_name: str           # Human-readable name for logs
    cost_per_1m_input: float    # USD per 1 million input tokens
    cost_per_1m_output: float   # USD per 1 million output tokens
    default_max_tokens: int     # Default max output tokens


# Model registry — update these when Anthropic releases new models or changes pricing
MODEL_REGISTRY: dict[ModelTier, ModelConfig] = {
    ModelTier.FAST: ModelConfig(
        model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        cost_per_1m_input=1.00,
        cost_per_1m_output=5.00,
        default_max_tokens=1024,
    ),
    ModelTier.STANDARD: ModelConfig(
        model_id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        cost_per_1m_input=3.00,
        cost_per_1m_output=15.00,
        default_max_tokens=4096,
    ),
    ModelTier.ADVANCED: ModelConfig(
        model_id="claude-opus-4-6",
        display_name="Claude Opus 4.6",
        cost_per_1m_input=15.00,
        cost_per_1m_output=75.00,
        default_max_tokens=4096,
    ),
}


# =============================================================================
# RESPONSE DATACLASS
# =============================================================================

@dataclass
class LLMResult:
    """
    Structured result from an LLM call.

    Every field here maps to a column in the AuditLog table,
    so logging is just a matter of passing this object.
    """

    text: str                   # The model's text response
    model_id: str               # Which model was used
    model_tier: ModelTier       # Which tier was requested
    tokens_input: int           # Input tokens consumed
    tokens_output: int          # Output tokens generated
    cost_usd: float             # Total cost of this call in USD
    latency_ms: int             # Wall-clock time in milliseconds
    is_mock: bool = False       # True if this was a dry-run response

    def __repr__(self) -> str:
        return (
            f"<LLMResult(model='{self.model_id}', "
            f"tokens={self.tokens_input}+{self.tokens_output}, "
            f"cost=${self.cost_usd:.6f}, latency={self.latency_ms}ms)>"
        )


# =============================================================================
# COST CALCULATOR
# =============================================================================

def calculate_cost(
    config: ModelConfig,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Calculate the cost of an LLM call in USD.

    Args:
        config: The model configuration with pricing.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Cost in USD (e.g., 0.000453).
    """
    input_cost = (input_tokens / 1_000_000) * config.cost_per_1m_input
    output_cost = (output_tokens / 1_000_000) * config.cost_per_1m_output
    return input_cost + output_cost


# =============================================================================
# MOCK RESPONSES (for dry-run mode)
# =============================================================================

# These mock responses are realistic enough to test the full pipeline
# without making API calls. Each tier returns a different style of response.
MOCK_RESPONSES: dict[ModelTier, str] = {
    ModelTier.FAST: (
        '{"score": 72, "eligible": true, '
        '"reasoning": "This tender matches our EHS software capabilities. '
        'Geography: US (supported). Budget: within range. '
        'Scope: SDS management software procurement. '
        'Domain match: high."}'
    ),
    ModelTier.STANDARD: (
        "Our company brings over a decade of experience in chemical safety "
        "and SDS management to this procurement. Our cloud-based platform "
        "currently serves 500+ organisations across manufacturing, construction, "
        "and pharmaceuticals. We provide GHS-compliant Safety Data Sheet management, "
        "automated regulatory reporting (OSHA Tier II, EPCRA/CERCLA), and mobile "
        "access via QR codes and dedicated apps. Our solution has demonstrated "
        "a 40% reduction in compliance audit preparation time for existing clients."
    ),
    ModelTier.ADVANCED: (
        "COMPLIANCE ANALYSIS:\n\n"
        "Requirement: ISO 27001 Information Security Management certification.\n"
        "Status: COMPLIANT\n"
        "Evidence: Our company holds ISO 27001:2022 certification (Certificate #IS-2025-0042), "
        "valid through March 2027. The certification covers our cloud infrastructure, "
        "data processing operations, and customer data handling procedures.\n\n"
        "Requirement: SOC 2 Type II audit report.\n"
        "Status: PARTIAL - Needs verification.\n"
        "Note: Our most recent SOC 2 Type II report covers the period ending June 2025. "
        "The tender requires a report no older than 12 months. Recommend escalation to "
        "confirm if a more recent audit is available or in progress."
    ),
}


# =============================================================================
# THE CLIENT
# =============================================================================

class MultiModelClient:
    """
    Multi-model Anthropic client with cost tracking.

    Handles:
    - Model routing based on tier (FAST/STANDARD/ADVANCED)
    - Token counting and cost calculation
    - Dry-run mode for testing without API keys
    - Structured logging of every call
    - Cumulative cost tracking across the session
    """

    def __init__(self, dry_run: bool | None = None) -> None:
        """
        Initialise the client.

        Args:
            dry_run: If True, return mock responses instead of calling the API.
                     If None (default), uses the DRY_RUN setting from .env.
        """
        self.dry_run = dry_run if dry_run is not None else settings.dry_run

        # Cumulative cost tracker for the session
        self._total_cost: float = 0.0
        self._total_calls: int = 0

        # Only create the real client if not in dry-run mode
        self._client: Anthropic | None = None
        if not self.dry_run:
            self._client = Anthropic(api_key=settings.anthropic_api_key)
            logger.info("llm_client_initialised", mode="live", models=[
                f"{tier.value}: {cfg.model_id}" for tier, cfg in MODEL_REGISTRY.items()
            ])
        else:
            logger.info("llm_client_initialised", mode="dry_run")

    def call(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.STANDARD,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResult:
        """
        Make an LLM call with automatic model routing and cost tracking.

        Args:
            prompt: The user message to send to the model.
            tier: Which model tier to use (FAST, STANDARD, ADVANCED).
            system: Optional system prompt that sets the model's behaviour.
            max_tokens: Maximum output tokens. If None, uses the tier's default.
            temperature: Sampling temperature. 0.0 = deterministic (best for scoring).
                         Higher values (0.3–0.7) add creativity (useful for drafting).

        Returns:
            LLMResult with the response text, token counts, cost, and latency.
        """
        config = MODEL_REGISTRY[tier]
        max_tokens = max_tokens or config.default_max_tokens

        if self.dry_run:
            return self._mock_call(tier, config)

        return self._real_call(prompt, tier, config, system, max_tokens, temperature)

    def _real_call(
        self,
        prompt: str,
        tier: ModelTier,
        config: ModelConfig,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResult:
        """Make a real API call to Anthropic."""
        assert self._client is not None, "Client not initialised — are you in dry_run mode?"

        start_time = time.monotonic()

        # Build the API request
        kwargs: dict = {
            "model": config.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Add system prompt if provided
        if system:
            kwargs["system"] = system

        # Make the call
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as e:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "llm_call_failed",
                tier=tier.value,
                model=config.model_id,
                error=str(e),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = int((time.monotonic() - start_time) * 1000)

        # Extract the text response
        text = response.content[0].text if response.content else ""

        # Get token counts from the API response
        tokens_input = response.usage.input_tokens
        tokens_output = response.usage.output_tokens

        # Calculate cost
        cost = calculate_cost(config, tokens_input, tokens_output)

        # Update cumulative tracking
        self._total_cost += cost
        self._total_calls += 1

        # Log the call
        logger.info(
            "llm_call_complete",
            tier=tier.value,
            model=config.model_id,
            tokens_in=tokens_input,
            tokens_out=tokens_output,
            cost_usd=f"${cost:.6f}",
            latency_ms=latency_ms,
            cumulative_cost=f"${self._total_cost:.4f}",
        )

        return LLMResult(
            text=text,
            model_id=config.model_id,
            model_tier=tier,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost,
            latency_ms=latency_ms,
            is_mock=False,
        )

    def _mock_call(
        self,
        tier: ModelTier,
        config: ModelConfig,
    ) -> LLMResult:
        """Return a mock response for dry-run mode."""
        # Simulate realistic token counts
        mock_token_counts = {
            ModelTier.FAST: (150, 80),       # Short classification calls
            ModelTier.STANDARD: (500, 400),  # Medium drafting calls
            ModelTier.ADVANCED: (800, 600),  # Longer compliance analysis
        }

        tokens_input, tokens_output = mock_token_counts[tier]
        cost = calculate_cost(config, tokens_input, tokens_output)

        self._total_cost += cost
        self._total_calls += 1

        logger.info(
            "llm_mock_call",
            tier=tier.value,
            model=config.model_id,
            tokens_in=tokens_input,
            tokens_out=tokens_output,
            cost_usd=f"${cost:.6f}",
        )

        return LLMResult(
            text=MOCK_RESPONSES[tier],
            model_id=config.model_id,
            model_tier=tier,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost,
            latency_ms=50,  # Simulated latency
            is_mock=True,
        )

    @property
    def total_cost(self) -> float:
        """Total cost of all calls in this session (USD)."""
        return self._total_cost

    @property
    def total_calls(self) -> int:
        """Total number of calls made in this session."""
        return self._total_calls

    def get_cost_summary(self) -> dict:
        """
        Returns a summary of costs for the current session.
        Useful for the audit log and cost tracking dashboard.
        """
        return {
            "total_calls": self._total_calls,
            "total_cost_usd": round(self._total_cost, 6),
            "models_available": {
                tier.value: config.model_id
                for tier, config in MODEL_REGISTRY.items()
            },
            "mode": "dry_run" if self.dry_run else "live",
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
# Import this everywhere: from src.utils.llm_client import llm_client
# It uses the DRY_RUN setting from .env automatically.

llm_client = MultiModelClient()