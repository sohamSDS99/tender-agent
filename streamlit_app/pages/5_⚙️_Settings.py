"""Settings page — Configure API keys, Slack, and agent parameters."""
import streamlit as st
import os

st.set_page_config(page_title="Settings — Tender Agent", page_icon="⚙️", layout="wide")

st.markdown("## ⚙️ Settings")
st.caption("Configure API keys, integrations, and agent behavior.")

st.divider()

# --- Agent Mode ---
st.markdown("### Agent Mode")
dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

if dry_run:
    st.warning(
        "**DRY RUN MODE** — The agent uses mock data and does not make real API calls. "
        "Set `DRY_RUN=false` in your `.env` file to enable production mode.",
        icon="🔸"
    )
else:
    st.success("**LIVE MODE** — The agent is making real API calls.", icon="🟢")

st.markdown("")

# --- API Keys ---
st.markdown("### API Keys")

tab1, tab2, tab3, tab4 = st.tabs(["🤖 Qwen (DashScope)", "🧭 Voyage AI", "💬 Slack", "📧 Email (SMTP)"])

with tab1:
    st.markdown("Qwen API (DashScope) for tender evaluation and drafting.")
    api_key = st.text_input(
        "DASHSCOPE_API_KEY",
        value=os.getenv("DASHSCOPE_API_KEY", ""),
        type="password",
        help="Get your key from dashscope.aliyuncs.com",
    )
    st.caption("Models used: Qwen3.5 Flash (eval), Qwen3.5 Plus (drafting), Qwen3 Max (compliance)")
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
        st.success("Key set for this session", icon="✅")

with tab2:
    st.markdown("Voyage AI for document embeddings (knowledge base search).")
    voyage_key = st.text_input(
        "VOYAGE_API_KEY",
        value=os.getenv("VOYAGE_API_KEY", ""),
        type="password",
        help="Get your key from dash.voyageai.com",
    )
    st.caption("Model: voyage-3-large (1024 dimensions)")
    if voyage_key:
        os.environ["VOYAGE_API_KEY"] = voyage_key
        st.success("Key set for this session", icon="✅")

with tab3:
    st.markdown("Slack integration for gap escalation and deadline alerts.")
    col1, col2 = st.columns(2)
    with col1:
        slack_token = st.text_input(
            "SLACK_BOT_TOKEN", value=os.getenv("SLACK_BOT_TOKEN", ""), type="password",
        )
    with col2:
        slack_channel = st.text_input(
            "SLACK_CHANNEL_ID", value=os.getenv("SLACK_CHANNEL_ID", ""),
        )
    st.caption("Create a Slack App at api.slack.com/apps with scopes: chat:write, channels:read, channels:history")
    if slack_token:
        os.environ["SLACK_BOT_TOKEN"] = slack_token
    if slack_channel:
        os.environ["SLACK_CHANNEL_ID"] = slack_channel

with tab4:
    st.markdown("SMTP configuration for email-based tender submissions.")
    c1, c2, c3 = st.columns(3)
    with c1:
        smtp_host = st.text_input("SMTP_HOST", value=os.getenv("SMTP_HOST", "smtp.gmail.com"))
        smtp_user = st.text_input("SMTP_USER", value=os.getenv("SMTP_USER", ""))
    with c2:
        smtp_port = st.text_input("SMTP_PORT", value=os.getenv("SMTP_PORT", "587"))
        smtp_pass = st.text_input("SMTP_PASSWORD", value=os.getenv("SMTP_PASSWORD", ""), type="password")
    with c3:
        smtp_from = st.text_input("SMTP_FROM", value=os.getenv("SMTP_FROM", ""))
    st.caption("For Gmail: use an App Password (Google Account → Security → App Passwords)")

st.markdown("")
st.divider()

# --- Evaluation Settings ---
st.markdown("### Evaluation Settings")
col1, col2 = st.columns(2)
with col1:
    threshold = st.number_input(
        "Eligibility Score Threshold",
        min_value=0, max_value=100, value=60,
        help="Tenders scoring below this are rejected. Default: 60/100."
    )
with col2:
    max_tenders = st.number_input(
        "Max Tenders Per Discovery Run",
        min_value=1, max_value=50, value=20,
        help="Limit how many tenders are processed per discovery cycle."
    )

st.markdown("")

# --- Budget Settings ---
st.markdown("### Budget Settings")
col1, col2 = st.columns(2)
with col1:
    budget = st.number_input(
        "Monthly LLM Budget ($)",
        min_value=0.0, max_value=1000.0, value=300.0, step=50.0,
        help="The cost dashboard will warn at 80% and alert at 100%."
    )
with col2:
    st.markdown("")
    st.markdown(f"""
    <div style="padding: 12px 16px; background: rgba(6, 182, 212, 0.05); 
         border-radius: 8px; border: 1px solid rgba(6, 182, 212, 0.1);">
        <div style="color: #94A3B8; font-size: 0.8rem;">Estimated cost per tender</div>
        <div style="color: #F1F5F9; font-size: 1.2rem; font-weight: 600;">~$0.15 — $0.25</div>
        <div style="color: #64748B; font-size: 0.75rem;">Qwen3.5 Flash eval + Qwen3.5 Plus draft + Qwen3.5 Plus gap check</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.divider()

# --- System Info ---
st.markdown("### System Information")
import platform
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Python:** {platform.python_version()}")
    st.markdown(f"**OS:** {platform.system()} {platform.machine()}")
    st.markdown(f"**DRY_RUN:** `{os.getenv('DRY_RUN', 'true')}`")
with col2:
    try:
        import langgraph
        st.markdown(f"**LangGraph:** installed")
    except ImportError:
        st.markdown("**LangGraph:** not found")
    try:
        import openai
        st.markdown(f"**OpenAI SDK (Qwen):** installed")
    except ImportError:
        st.markdown("**OpenAI SDK (Qwen):** not found")
    st.markdown(f"**Project Path:** `{os.getcwd()}`")