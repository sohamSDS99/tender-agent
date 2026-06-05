"""
nexus-sdk — Python client for the Nexus Agent Management System.

Quick start::

    from nexus_sdk import NexusClient, NexusMiddleware
    from nexus_sdk.types import AgentConfig

    client = NexusClient(base_url="http://localhost:3000")
    client.register(AgentConfig(
        name="my-agent",
        display_name="My Agent",
        version="1.0.0",
    ))
    client.heartbeat()
    client.track("latency", 142.5, metadata={"node": "evaluate"})
"""

from .client import NexusClient
from .middleware import NexusMiddleware
from .types import (
    AgentConfig,
    ChatMessage,
    HeartbeatPayload,
    IngestResponse,
    MetricEvent,
    RegisterResponse,
)

__all__ = [
    "NexusClient",
    "NexusMiddleware",
    "AgentConfig",
    "ChatMessage",
    "HeartbeatPayload",
    "IngestResponse",
    "MetricEvent",
    "RegisterResponse",
]

__version__ = "0.1.0"
