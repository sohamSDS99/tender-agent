"""
Tests the multi-model LLM client.

Runs in dry-run mode so no API key or real API calls are needed.
Verifies all 3 tiers, cost calculation, and the cost summary.

Usage:
    python scripts/test_llm_client.py
"""

from src.utils.llm_client import (
    MultiModelClient,
    ModelTier,
    ModelConfig,
    MODEL_REGISTRY,
    calculate_cost,
    LLMResult,
)
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def test_cost_calculation() -> None:
    """Test that cost calculation is mathematically correct."""
    print("\n" + "=" * 60)
    print("  TEST 1: Cost Calculation")
    print("=" * 60 + "\n")

    config = MODEL_REGISTRY[ModelTier.FAST]  # Haiku: $1.00/$5.00 per 1M

    # 1000 input tokens + 500 output tokens with Haiku
    cost = calculate_cost(config, input_tokens=1000, output_tokens=500)
    expected = (1000 / 1_000_000) * 1.00 + (500 / 1_000_000) * 5.00
    assert abs(cost - expected) < 1e-10, f"Expected {expected}, got {cost}"
    print(f"  Haiku: 1000 in + 500 out = ${cost:.6f} (expected ${expected:.6f})")

    config = MODEL_REGISTRY[ModelTier.STANDARD]  # Sonnet: $3.00/$15.00 per 1M
    cost = calculate_cost(config, input_tokens=500, output_tokens=400)
    expected = (500 / 1_000_000) * 3.00 + (400 / 1_000_000) * 15.00
    assert abs(cost - expected) < 1e-10
    print(f"  Sonnet: 500 in + 400 out = ${cost:.6f}")

    config = MODEL_REGISTRY[ModelTier.ADVANCED]  # Opus: $15.00/$75.00 per 1M
    cost = calculate_cost(config, input_tokens=800, output_tokens=600)
    expected = (800 / 1_000_000) * 15.00 + (600 / 1_000_000) * 75.00
    assert abs(cost - expected) < 1e-10
    print(f"  Opus: 800 in + 600 out = ${cost:.6f}")

    print("\n  TEST 1 PASSED: Cost calculation is correct.\n")


def test_dry_run_calls() -> None:
    """Test all 3 tiers in dry-run mode."""
    print("=" * 60)
    print("  TEST 2: Dry-Run Calls (all 3 tiers)")
    print("=" * 60 + "\n")

    # Force dry_run=True regardless of .env setting
    client = MultiModelClient(dry_run=True)

    # Test FAST tier (Haiku)
    result = client.call(
        prompt="Is this tender relevant to EHS software?",
        tier=ModelTier.FAST,
        system="You are a classifier.",
    )
    assert isinstance(result, LLMResult)
    assert result.is_mock is True
    assert result.model_id == "claude-haiku-4-5-20251001"
    assert result.tokens_input > 0
    assert result.cost_usd > 0
    assert len(result.text) > 0
    print(f"  FAST (Haiku):     {result.model_id}")
    print(f"    Response: {result.text[:80]}...")
    print(f"    Cost: ${result.cost_usd:.6f}")

    # Test STANDARD tier (Sonnet)
    result = client.call(
        prompt="Draft the executive summary.",
        tier=ModelTier.STANDARD,
    )
    assert result.model_id == "claude-sonnet-4-6"
    assert result.is_mock is True
    print(f"\n  STANDARD (Sonnet): {result.model_id}")
    print(f"    Response: {result.text[:80]}...")
    print(f"    Cost: ${result.cost_usd:.6f}")

    # Test ADVANCED tier (Opus)
    result = client.call(
        prompt="Analyse compliance requirements.",
        tier=ModelTier.ADVANCED,
    )
    assert result.model_id == "claude-opus-4-6"
    assert result.is_mock is True
    print(f"\n  ADVANCED (Opus):  {result.model_id}")
    print(f"    Response: {result.text[:80]}...")
    print(f"    Cost: ${result.cost_usd:.6f}")

    # Verify cumulative tracking
    assert client.total_calls == 3
    assert client.total_cost > 0
    print(f"\n  Total calls: {client.total_calls}")
    print(f"  Total cost:  ${client.total_cost:.6f}")

    print("\n  TEST 2 PASSED: All 3 tiers return correct mock responses.\n")


def test_cost_summary() -> None:
    """Test the cost summary output."""
    print("=" * 60)
    print("  TEST 3: Cost Summary")
    print("=" * 60 + "\n")

    client = MultiModelClient(dry_run=True)

    # Make a few calls
    client.call("test1", tier=ModelTier.FAST)
    client.call("test2", tier=ModelTier.STANDARD)
    client.call("test3", tier=ModelTier.ADVANCED)
    client.call("test4", tier=ModelTier.FAST)

    summary = client.get_cost_summary()

    assert summary["total_calls"] == 4
    assert summary["total_cost_usd"] > 0
    assert summary["mode"] == "dry_run"
    assert "fast" in summary["models_available"]
    assert "standard" in summary["models_available"]
    assert "advanced" in summary["models_available"]

    print(f"  Summary: {summary}")

    print("\n  TEST 3 PASSED: Cost summary is correct.\n")


def test_model_registry() -> None:
    """Verify all models are properly configured."""
    print("=" * 60)
    print("  TEST 4: Model Registry")
    print("=" * 60 + "\n")

    for tier in ModelTier:
        assert tier in MODEL_REGISTRY, f"Missing model config for {tier}"
        config = MODEL_REGISTRY[tier]
        assert config.model_id, f"Empty model_id for {tier}"
        assert config.cost_per_1m_input > 0, f"Invalid input cost for {tier}"
        assert config.cost_per_1m_output > 0, f"Invalid output cost for {tier}"
        assert config.default_max_tokens > 0, f"Invalid max_tokens for {tier}"
        print(f"  {tier.value:12s} -> {config.model_id:35s} (${config.cost_per_1m_input}/{config.cost_per_1m_output} per 1M)")

    print("\n  TEST 4 PASSED: All models properly configured.\n")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  MULTI-MODEL LLM CLIENT — VERIFICATION SUITE")
    print("#" * 60)

    test_cost_calculation()
    test_dry_run_calls()
    test_cost_summary()
    test_model_registry()

    print("=" * 60)
    print("  ALL 4 TESTS PASSED")
    print("  Multi-model client with cost tracking verified")
    print("=" * 60)
    print()