# AI Tender Applying Agent

Autonomous AI agent that discovers government and enterprise tenders, evaluates them against company capabilities, drafts complete tender responses, and submits them to procurement portals.

## Architecture

- **Orchestration:** LangGraph 1.1 StateGraph with PostgreSQL checkpointing
- **LLMs:** Claude Haiku 4.5 (scoring), Sonnet 4.6 (drafting), Opus 4.6 (compliance)
- **Vector DB:** pgvector on PostgreSQL
- **Embeddings:** Voyage AI voyage-3-large

## Setup

1. Clone this repo
2. Copy `.env.example` to `.env` and fill in your API keys
3. Create a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
4. Install dependencies: `pip install -e ".[dev]"`
5. Set up PostgreSQL (see docs/step-02-database.md)

## Project Structure

See `docs/` for detailed architecture documentation.

## Status

🚧 Under active development. See the implementation roadmap in `docs/`.
