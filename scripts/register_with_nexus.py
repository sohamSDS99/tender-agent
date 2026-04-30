"""
Register the tender-agent with the Nexus AMS platform.

Run this ONCE to register (or re-register) the agent.
After registration, the agent appears in the AMS dashboard.

Usage:
    cd ~/Desktop/tender-agent
    source .venv/bin/activate
    python scripts/register_with_nexus.py
"""

import os
import sys
from pathlib import Path

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from nexus_sdk import NexusClient, AgentConfig

def main() -> None:
    nexus_url = os.getenv("NEXUS_AMS_URL", "http://localhost:3000")
    print(f"Connecting to Nexus AMS at: {nexus_url}")

    client = NexusClient(base_url=nexus_url)

    config = AgentConfig(
        name="tender-agent",
        display_name="Tender Agent",
        description=(
            "Discovers and evaluates government tenders related to Safety Data Sheet (SDS) "
            "management from SAM.gov and other procurement portals. Uses SERP API and "
            "LLM-powered evaluation to find, score, draft responses to, and submit "
            "relevant tenders automatically."
        ),
        version="1.0.0",
        python_version="3.11",
        langgraph_version="0.4.0",
        llm_provider="openrouter",
        llm_models={
            "qwen/qwen3-9b": {
                "provider": "openrouter",
                "context_window": 131072,
                "purpose": "primary reasoning and evaluation",
            },
        },
        llm_pricing={
            "qwen/qwen3-9b": {
                "input_cost_per_million": 0.20,
                "output_cost_per_million": 0.60,
            },
        },
        embedding_model="voyage-3-large",
        embedding_dimensions=1024,
        state_fields_count=40,
        node_names=[
            "discover",
            "evaluate",
            "retrieve_draft",
            "gap_check",
            "slack_escalate",
            "assemble",
            "submit",
        ],
        tools=[
            "sam_gov_scraper",
            "email_monitor",
            "voyage_embedder",
            "template_engine",
            "slack_client",
            "playwright_submitter",
        ],
        health_endpoint="http://localhost:8100/health",
        slack_channels=["#tender-alerts"],
        env_vars_count=23,
        dry_run=True,
        budget_monthly_usd=50.0,
        tags=["production", "government-tenders", "sds", "ehs", "procurement"],
        changelog="Initial registration with Nexus AMS via nexus-sdk",
    )

    print("Registering tender-agent...")
    try:
        response = client.register(config)
        print("")
        print("=" * 60)
        print("  REGISTRATION SUCCESSFUL")
        print("=" * 60)
        print(f"  Agent ID:   {response.agent_id}")
        print(f"  Name:       {response.name}")
        print(f"  Version:    {response.version}")
        print("=" * 60)
        print("")
        print("The tender-agent is now visible in the AMS dashboard.")
        print(f"Open http://localhost:3000/agents to see it.")
    except Exception as exc:
        print(f"Registration failed: {exc}")
        print("")
        print("Make sure the AMS is running: cd ~/Desktop/nexus-ams-main && pnpm dev")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
