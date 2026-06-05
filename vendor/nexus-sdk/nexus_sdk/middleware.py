"""
NexusMiddleware — Instruments a LangGraph StateGraph with Nexus tracking.

Wraps each node's entry and exit with track() calls so you get:
- Per-node latency
- Task completion events
- Error rates

Usage::

    from nexus_sdk import NexusClient, NexusMiddleware

    client = NexusClient(base_url="http://localhost:3000")
    builder = StateGraph(MyState)
    # ... add_node calls ...
    instrumented = NexusMiddleware(builder, client)
    graph = instrumented.compile()
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from .client import NexusClient

logger = logging.getLogger(__name__)


class NexusMiddleware:
    """
    Wraps a LangGraph StateGraph to automatically instrument all nodes
    with Nexus metric tracking.

    The graph builder is wrapped at construction time. Call compile()
    to get the runnable LangGraph.
    """

    def __init__(self, graph_builder: Any, client: NexusClient) -> None:
        """
        Parameters
        ----------
        graph_builder:
            A LangGraph StateGraph (or CompiledGraph) instance.
        client:
            An initialized NexusClient (register() should already have been called).
        """
        self._builder = graph_builder
        self._client = client
        self._instrumented_nodes: list[str] = []
        self._instrument_all_nodes()

    def _instrument_all_nodes(self) -> None:
        """
        Walk the graph's node registry and wrap each node function
        with entry/exit tracking.
        """
        # LangGraph StateGraph stores nodes in _nodes dict
        nodes: dict[str, Any] = getattr(self._builder, "_nodes", {})

        for node_name, node_spec in nodes.items():
            # node_spec is typically a NodeSpec namedtuple with a .runnable field
            original_runnable = getattr(node_spec, "runnable", None)
            if original_runnable is None:
                continue

            wrapped = self._wrap_node(node_name, original_runnable)

            # Replace the runnable in the node spec
            try:
                object.__setattr__(node_spec, "runnable", wrapped)
                self._instrumented_nodes.append(node_name)
                logger.debug("Instrumented node: %s", node_name)
            except (AttributeError, TypeError):
                logger.warning("Could not instrument node %s", node_name)

        if self._instrumented_nodes:
            logger.info(
                "NexusMiddleware instrumented %d nodes: %s",
                len(self._instrumented_nodes),
                ", ".join(self._instrumented_nodes),
            )

    def _wrap_node(self, node_name: str, original: Any) -> Callable[..., Any]:
        """Return a wrapper that tracks entry/exit/error for a node."""
        client = self._client

        def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            logger.debug("[nexus] → %s", node_name)

            try:
                result = original(state, *args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000

                client.track(
                    "latency",
                    elapsed_ms,
                    metadata={"node": node_name, "status": "success"},
                )
                client.track(
                    "task_completion",
                    1.0,
                    metadata={"node": node_name},
                )
                logger.debug("[nexus] ✓ %s (%.1fms)", node_name, elapsed_ms)
                return result

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                client.track(
                    "error_rate",
                    1.0,
                    metadata={
                        "node": node_name,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:200],
                    },
                )
                client.track(
                    "latency",
                    elapsed_ms,
                    metadata={"node": node_name, "status": "error"},
                )
                logger.debug("[nexus] ✗ %s — %s (%.1fms)", node_name, exc, elapsed_ms)
                raise

        wrapper.__name__ = f"nexus_{node_name}"
        return wrapper

    def compile(self, **kwargs: Any) -> Any:
        """Compile the instrumented graph, wrapping invoke/ainvoke with task_start/task_end events."""
        compiled = self._builder.compile(**kwargs)
        client = self._client
        agent_name = getattr(client, "agent_name", "unknown")

        original_invoke = compiled.invoke

        def tracked_invoke(state: Any, config: Any = None, **kw: Any) -> Any:
            task_id = f"{agent_name}-{int(time.time() * 1000)}"
            start = time.perf_counter()
            client.track("task_start", 1.0, metadata={"agent_name": agent_name, "task_id": task_id, "task_type": "graph_run"})
            try:
                result = original_invoke(state, config, **kw)
                duration_ms = (time.perf_counter() - start) * 1000
                client.track("task_end", 1.0, metadata={"agent_name": agent_name, "task_id": task_id, "task_type": "graph_run", "duration_ms": round(duration_ms, 2), "status": "success"})
                return result
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000
                client.track("task_end", 0.0, metadata={"agent_name": agent_name, "task_id": task_id, "task_type": "graph_run", "duration_ms": round(duration_ms, 2), "status": "failure", "error": str(exc)[:200]})
                raise

        compiled.invoke = tracked_invoke
        return compiled

    @property
    def instrumented_nodes(self) -> list[str]:
        """Names of nodes that were successfully instrumented."""
        return list(self._instrumented_nodes)
