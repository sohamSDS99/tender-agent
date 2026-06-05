"""
mock_agent.py — Example agent using the nexus-sdk.

Demonstrates:
- Client initialization
- Agent registration
- Heartbeat loop (in a thread)
- Emitting 10 fake metrics

Run with:
    python examples/mock_agent.py
"""

from __future__ import annotations

import logging
import random
import sys
import threading
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from nexus_sdk import NexusClient
from nexus_sdk.types import AgentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("mock-agent")

NEXUS_URL = "http://localhost:3000"
NODE_NAMES = ["ingest", "process", "validate", "output"]


def main() -> None:
    logger.info("Starting mock-agent...")

    client = NexusClient(base_url=NEXUS_URL)

    # 1. Register the agent
    try:
        resp = client.register(
            AgentConfig(
                name="mock-agent",
                display_name="Mock Agent",
                description="Development mock agent for testing the nexus-sdk",
                version="0.1.0",
                python_version="3.12",
                llm_provider="openai",
                node_names=NODE_NAMES,
                state_fields_count=12,
                tags=["mock", "development"],
                changelog="Initial mock agent for SDK validation",
            )
        )
        logger.info("Registered! agent_id=%s", resp.agent_id)
    except Exception as e:
        logger.error("Registration failed: %s", e)
        logger.info("Continuing with mock agent_id for metric demo...")
        # Use a hardcoded ID for demo if server isn't up
        client.agent_id = "00000000-0000-0000-0000-000000000000"

    # 2. Start heartbeat in a background thread
    def heartbeat_thread() -> None:
        for i in range(5):
            try:
                client.heartbeat(status="running")
                logger.info("Heartbeat #%d OK", i + 1)
            except Exception as e:
                logger.warning("Heartbeat failed: %s", e)
            time.sleep(10)

    hb = threading.Thread(target=heartbeat_thread, daemon=True)
    hb.start()

    # 3. Emit 10 fake metrics
    logger.info("Emitting 10 fake metrics...")

    for i in range(10):
        node = random.choice(NODE_NAMES)

        # Latency metric
        latency_ms = random.gauss(150, 40)
        client.track(
            "latency",
            max(10, latency_ms),
            metadata={"node": node, "iteration": i},
        )

        # Token usage metric
        tokens = random.randint(100, 2000)
        client.track(
            "token_usage",
            float(tokens),
            metadata={"node": node, "model": "gpt-4o"},
        )

        # Cost metric (rough estimate: $0.01 per 1k tokens)
        cost = tokens * 0.00001
        client.track("cost", cost, metadata={"node": node})

        logger.info(
            "Metric batch %d/10: node=%s latency=%.1fms tokens=%d cost=$%.5f",
            i + 1,
            node,
            latency_ms,
            tokens,
            cost,
        )

        time.sleep(0.5)

    # 4. Force-flush remaining buffered metrics
    try:
        result = client.flush()
        logger.info("Flushed %d metrics to Nexus", result.inserted)
    except Exception as e:
        logger.warning("Flush failed (server may not be running): %s", e)

    # 5. Wait for heartbeat thread
    hb.join(timeout=2)

    # 6. Clean up
    client.close()
    logger.info("Mock agent finished. Check the Nexus dashboard!")


if __name__ == "__main__":
    main()
