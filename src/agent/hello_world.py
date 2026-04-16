"""
LangGraph Hello World — minimal 3-node graph with PostgreSQL checkpointing.

This is a proof-of-concept that verifies:
1. LangGraph StateGraph compiles and runs
2. State flows correctly between nodes
3. Conditional edges work (branching based on score)
4. PostgreSQL checkpointing saves and restores state

Run with:
    python -m src.agent.hello_world
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


# =============================================================================
# 1. DEFINE THE STATE
# =============================================================================
# This is the shared dictionary that flows through the graph.
# Every node can read any field and return updates to any field.

class HelloState(TypedDict):
    """
    Minimal state for the hello world graph.

    In the real agent, this becomes TenderState with 20+ fields.
    Here we keep it simple to focus on the mechanics.
    """

    title: str          # Tender title
    source: str         # Where the tender came from
    score: int          # Evaluation score (0-100)
    status: str         # Current status: "new", "accepted", "rejected"
    processing_log: list[str]  # Log of which nodes ran (for verification)


# =============================================================================
# 2. DEFINE THE NODES
# =============================================================================
# Each node is a plain Python function that takes state and returns updates.
# LangGraph merges the returned dict into the state automatically.

def intake_node(state: HelloState) -> dict:
    """
    Simulates the Discover node.

    In the real agent, this would scrape tender portals and extract metadata.
    Here, it just sets a title and source to prove state flows.
    """
    logger.info("node_running", node="intake")

    return {
        "title": "EHS Software Procurement - City of Denver",
        "source": "sam_gov",
        "status": "new",
        "processing_log": state.get("processing_log", []) + ["intake"],
    }


def score_node(state: HelloState) -> dict:
    """
    Simulates the Evaluate node.

    In the real agent, this would call Claude Haiku to score the tender.
    Here, we hardcode a score to test conditional routing.
    """
    logger.info("node_running", node="score", title=state["title"])

    # Hardcoded score for testing. Change this to < 60 to test the reject path.
    calculated_score = 75

    return {
        "score": calculated_score,
        "processing_log": state.get("processing_log", []) + ["score"],
    }


def accept_node(state: HelloState) -> dict:
    """
    Simulates advancing a tender to the Draft stage.
    """
    logger.info("node_running", node="accept", score=state["score"])

    return {
        "status": "accepted",
        "processing_log": state.get("processing_log", []) + ["accept"],
    }


def reject_node(state: HelloState) -> dict:
    """
    Simulates archiving a rejected tender.
    """
    logger.info("node_running", node="reject", score=state["score"])

    return {
        "status": "rejected",
        "processing_log": state.get("processing_log", []) + ["reject"],
    }


# =============================================================================
# 3. DEFINE THE ROUTING FUNCTION
# =============================================================================
# This function decides which path to take based on the current state.
# It returns the NAME of the next node as a string.

def decide_route(state: HelloState) -> str:
    """
    Conditional edge: routes based on score threshold.

    In the real agent, this threshold comes from settings.tender_score_threshold.
    """
    threshold = 60

    if state["score"] >= threshold:
        logger.info("routing_decision", decision="accept", score=state["score"])
        return "accept"
    else:
        logger.info("routing_decision", decision="reject", score=state["score"])
        return "reject"


# =============================================================================
# 4. BUILD THE GRAPH
# =============================================================================

def build_hello_graph() -> StateGraph:
    """
    Constructs the hello world graph.

    Graph structure:
        START → intake → score → decide_route → accept → END
                                              → reject → END

    Returns the builder (not compiled). The caller compiles it with
    or without a checkpointer depending on the use case.
    """
    # Create the graph builder with our state type
    builder = StateGraph(HelloState)

    # Add nodes — each string name maps to a function
    builder.add_node("intake", intake_node)
    builder.add_node("score", score_node)
    builder.add_node("accept", accept_node)
    builder.add_node("reject", reject_node)

    # Add edges — define the flow
    # START → intake (first node to run)
    builder.add_edge(START, "intake")

    # intake → score (always)
    builder.add_edge("intake", "score")

    # score → conditional routing based on decide_route()
    builder.add_conditional_edges(
        "score",                    # After this node...
        decide_route,               # ...call this function to decide...
        {                           # ...and route based on the return value:
            "accept": "accept",     # If decide_route returns "accept", go to accept node
            "reject": "reject",     # If decide_route returns "reject", go to reject node
        },
    )

    # accept → END
    builder.add_edge("accept", END)

    # reject → END
    builder.add_edge("reject", END)

    return builder


# =============================================================================
# 5. RUN WITHOUT CHECKPOINTING (basic test)
# =============================================================================

def run_without_checkpointing() -> None:
    """
    Runs the graph without checkpointing.
    This is the simplest possible test — just prove the graph compiles and runs.
    """
    print("\n" + "=" * 60)
    print("  TEST 1: Running graph WITHOUT checkpointing")
    print("=" * 60 + "\n")

    builder = build_hello_graph()
    graph = builder.compile()  # No checkpointer — state lives only in memory

    # Invoke the graph with empty initial state
    result = graph.invoke({
        "title": "",
        "source": "",
        "score": 0,
        "status": "",
        "processing_log": [],
    })

    print(f"  Title:          {result['title']}")
    print(f"  Source:         {result['source']}")
    print(f"  Score:          {result['score']}")
    print(f"  Status:         {result['status']}")
    print(f"  Processing log: {result['processing_log']}")

    # Verify
    assert result["title"] == "EHS Software Procurement - City of Denver"
    assert result["source"] == "sam_gov"
    assert result["score"] == 75
    assert result["status"] == "accepted"
    assert result["processing_log"] == ["intake", "score", "accept"]

    print("\n  TEST 1 PASSED: Graph compiles, runs, and produces correct output.\n")


# =============================================================================
# 6. RUN WITH POSTGRESQL CHECKPOINTING
# =============================================================================

def run_with_checkpointing() -> None:
    """
    Runs the graph with PostgreSQL checkpointing.

    This proves that:
    - LangGraph can connect to our PostgreSQL database
    - Checkpoints are saved after each node
    - State can be retrieved after the graph completes
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    from src.utils.config import settings

    # The checkpointer needs a raw PostgreSQL connection string (not SQLAlchemy format).
    # Our .env has: postgresql+psycopg://user:pass@host:port/db
    # The checkpointer needs: postgresql://user:pass@host:port/db
    db_uri = settings.database_url.replace("+psycopg", "")

    print("\n" + "=" * 60)
    print("  TEST 2: Running graph WITH PostgreSQL checkpointing")
    print("=" * 60 + "\n")

    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
        # .setup() creates the checkpoint tables in PostgreSQL.
        # Safe to call multiple times — it uses IF NOT EXISTS.
        checkpointer.setup()
        logger.info("checkpointer_ready", backend="postgresql")

        builder = build_hello_graph()
        graph = builder.compile(checkpointer=checkpointer)

        # Every graph invocation needs a thread_id.
        # This identifies this specific execution so we can resume it later.
        config = {"configurable": {"thread_id": "hello-world-test-001"}}

        result = graph.invoke(
            {
                "title": "",
                "source": "",
                "score": 0,
                "status": "",
                "processing_log": [],
            },
            config=config,
        )

        print(f"  Title:          {result['title']}")
        print(f"  Source:         {result['source']}")
        print(f"  Score:          {result['score']}")
        print(f"  Status:         {result['status']}")
        print(f"  Processing log: {result['processing_log']}")

        # Verify the results
        assert result["status"] == "accepted"
        assert result["processing_log"] == ["intake", "score", "accept"]
        print("\n  Graph execution correct.")

        # Now prove the checkpoint was saved by retrieving the state
        saved_state = graph.get_state(config)
        assert saved_state.values["status"] == "accepted"
        assert saved_state.values["score"] == 75
        print("  Checkpoint retrieved successfully from PostgreSQL.")
        print(f"  Saved state status: {saved_state.values['status']}")
        print(f"  Saved state score:  {saved_state.values['score']}")

    print("\n  TEST 2 PASSED: Checkpointing to PostgreSQL works.\n")


# =============================================================================
# 7. RUN THE REJECT PATH (test conditional routing)
# =============================================================================

def run_reject_path() -> None:
    """
    Tests the reject path by temporarily modifying the score.

    We create a modified score node that returns a low score,
    rebuild the graph, and verify it routes to reject.
    """
    print("\n" + "=" * 60)
    print("  TEST 3: Testing the REJECT path (score < 60)")
    print("=" * 60 + "\n")

    def low_score_node(state: HelloState) -> dict:
        """Score node that returns a failing score."""
        return {
            "score": 45,
            "processing_log": state.get("processing_log", []) + ["score"],
        }

    # Build a custom graph with the low-score node
    builder = StateGraph(HelloState)
    builder.add_node("intake", intake_node)
    builder.add_node("score", low_score_node)  # Using the low-score version
    builder.add_node("accept", accept_node)
    builder.add_node("reject", reject_node)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "score")
    builder.add_conditional_edges(
        "score",
        decide_route,
        {"accept": "accept", "reject": "reject"},
    )
    builder.add_edge("accept", END)
    builder.add_edge("reject", END)

    graph = builder.compile()

    result = graph.invoke({
        "title": "",
        "source": "",
        "score": 0,
        "status": "",
        "processing_log": [],
    })

    print(f"  Score:          {result['score']}")
    print(f"  Status:         {result['status']}")
    print(f"  Processing log: {result['processing_log']}")

    assert result["score"] == 45
    assert result["status"] == "rejected"
    assert result["processing_log"] == ["intake", "score", "reject"]

    print("\n  TEST 3 PASSED: Conditional routing to reject path works.\n")


# =============================================================================
# MAIN — RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  LANGGRAPH HELLO WORLD — VERIFICATION SUITE")
    print("#" * 60)

    run_without_checkpointing()
    run_with_checkpointing()
    run_reject_path()

    print("=" * 60)
    print("  ALL 3 TESTS PASSED")
    print("  LangGraph StateGraph + PostgreSQL checkpointing verified")
    print("=" * 60)
    print()