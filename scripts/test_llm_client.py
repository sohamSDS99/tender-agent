"""
Tests the multi-model LLM client (Qwen API via OpenAI-compatible SDK).

Runs in dry-run mode so no API key or real API calls are needed.
Verifies all 3 tiers, cost calculation, and the cost summary.

Usage:
    python scripts/test_llm_client.py
"""

from src.utils.llm_client import (
    MultiModelClient,
    ModelTier,
    LLMResponse,
    LLMUsage,
    MODEL_COSTS,
)
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def test_cost_calculation() -> None:
    """Test that cost calculation is mathematically correct."""
    print("\n" + "=" * 60)
    print("  TEST 1: Cost Calculation")
    print("=" * 60 + "\n")

    # qwen3.5-flash: $0.07/$0.26 per 1M
    cost = MultiModelClient._calculate_cost("qwen3.5-flash", input_tokens=1000, output_tokens=500)
    expected = (1000 / 1_000_000) * 0.07 + (500 / 1_000_000) * 0.26
    assert abs(cost - expected) < 1e-10, f"Expected {expected}, got {cost}"
    print(f"  qwen3.5-flash: 1000 in + 500 out = ${cost:.6f} (expected ${expected:.6f})")

    # qwen3.5-plus: $0.26/$1.56 per 1M
    cost = MultiModelClient._calculate_cost("qwen3.5-plus", input_tokens=500, output_tokens=400)
    expected = (500 / 1_000_000) * 0.26 + (400 / 1_000_000) * 1.56
    assert abs(cost - expected) < 1e-10
    print(f"  qwen3.5-plus:  500 in + 400 out = ${cost:.6f}")

    # qwen3-max: $0.78/$3.90 per 1M
    cost = MultiModelClient._calculate_cost("qwen3-max", input_tokens=800, output_tokens=600)
    expected = (800 / 1_000_000) * 0.78 + (600 / 1_000_000) * 3.90
    assert abs(cost - expected) < 1e-10
    print(f"  qwen3-max:     800 in + 600 out = ${cost:.6f}")

    print("\n  TEST 1 PASSED: Cost calculation is correct.\n")


def test_dry_run_calls() -> None:
    """Test all 3 tiers in dry-run mode."""
    print("=" * 60)
    print("  TEST 2: Dry-Run Calls (all 3 tiers)")
    print("=" * 60 + "\n")

    # Force dry_run=True regardless of .env setting
    client = MultiModelClient(dry_run=True)

    # Test FAST tier (qwen3.5-flash)
    result = client.complete(
        prompt="Is this tender relevant to EHS software?",
        tier=ModelTier.FAST,
        system_prompt="You are a classifier.",
    )
    assert isinstance(result, LLMResponse)
    assert result.model == "qwen3.5-flash"
    assert result.usage.input_tokens > 0
    assert result.usage.cost_usd > 0
    assert len(result.content) > 0
    assert "DRY-RUN" in result.content
    print(f"  FAST (qwen3.5-flash):     {result.model}")
    print(f"    Response: {result.content[:80]}...")
    print(f"    Cost: ${result.usage.cost_usd:.6f}")

    # Test STANDARD tier (qwen3.5-plus)
    result = client.complete(
        prompt="Draft the executive summary.",
        tier=ModelTier.STANDARD,
    )
    assert result.model == "qwen3.5-plus"
    print(f"\n  STANDARD (qwen3.5-plus):  {result.model}")
    print(f"    Response: {result.content[:80]}...")
    print(f"    Cost: ${result.usage.cost_usd:.6f}")

    # Test ADVANCED tier (qwen3-max)
    result = client.complete(
        prompt="Analyse compliance requirements.",
        tier=ModelTier.ADVANCED,
    )
    assert result.model == "qwen3-max"
    print(f"\n  ADVANCED (qwen3-max):     {result.model}")
    print(f"    Response: {result.content[:80]}...")
    print(f"    Cost: ${result.usage.cost_usd:.6f}")

    # Verify cumulative tracking
    assert client.call_count == 3
    assert client.total_cost_usd > 0
    print(f"\n  Total calls: {client.call_count}")
    print(f"  Total cost:  ${client.total_cost_usd:.6f}")

    print("\n  TEST 2 PASSED: All 3 tiers return correct mock responses.\n")


def test_cost_summary() -> None:
    """Test the cost summary output."""
    print("=" * 60)
    print("  TEST 3: Cost Summary")
    print("=" * 60 + "\n")

    client = MultiModelClient(dry_run=True)

    # Make a few calls
    client.complete("test1", tier=ModelTier.FAST)
    client.complete("test2", tier=ModelTier.STANDARD)
    client.complete("test3", tier=ModelTier.ADVANCED)
    client.complete("test4", tier=ModelTier.FAST)

    summary = client.get_cost_summary()

    assert summary["call_count"] == 4
    assert summary["total_cost_usd"] > 0
    assert summary["total_input_tokens"] > 0
    assert summary["total_output_tokens"] > 0

    print(f"  Summary: {summary}")

    print("\n  TEST 3 PASSED: Cost summary is correct.\n")


def test_model_tiers() -> None:
    """Verify all model tiers map to correct Qwen model strings."""
    print("=" * 60)
    print("  TEST 4: Model Tiers")
    print("=" * 60 + "\n")

    expected = {
        ModelTier.FAST: "qwen3.5-flash",
        ModelTier.STANDARD: "qwen3.5-plus",
        ModelTier.ADVANCED: "qwen3-max",
    }

    for tier, model_id in expected.items():
        assert tier.value == model_id, f"Expected {model_id}, got {tier.value}"
        assert model_id in MODEL_COSTS, f"Missing pricing for {model_id}"
        costs = MODEL_COSTS[model_id]
        assert costs["input"] > 0
        assert costs["output"] > 0
        print(f"  {tier.name:12s} -> {model_id:20s} (${costs['input']}/{costs['output']} per 1M)")

    print("\n  TEST 4 PASSED: All model tiers correctly configured.\n")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  MULTI-MODEL LLM CLIENT (QWEN) — VERIFICATION SUITE")
    print("#" * 60)

    test_cost_calculation()
    test_dry_run_calls()
    test_cost_summary()
    test_model_tiers()

    print("=" * 60)
    print("  ALL 4 TESTS PASSED")
    print("  Multi-model Qwen client with cost tracking verified")
    print("=" * 60)
    print()
