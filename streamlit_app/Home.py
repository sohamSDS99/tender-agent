"""
AI Tender Agent — Command Center
Premium Streamlit dashboard for managing the autonomous tender pipeline.
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Tender Agent — Command Center",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp { font-family: 'DM Sans', sans-serif; }
    
    /* Hide default header */
    header[data-testid="stHeader"] { background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(12px); }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1A1F3A 100%);
        border-right: 1px solid rgba(6, 182, 212, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 { 
        font-size: 1.1rem; font-weight: 700; letter-spacing: 0.05em; 
        color: #06B6D4; text-transform: uppercase; 
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E293B 0%, #1A2332 100%);
        border: 1px solid rgba(6, 182, 212, 0.12);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(6, 182, 212, 0.15);
        border-color: rgba(6, 182, 212, 0.3);
    }
    div[data-testid="stMetric"] label { 
        color: #94A3B8 !important; font-size: 0.8rem !important; 
        text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { 
        color: #F1F5F9 !important; font-size: 1.8rem !important; font-weight: 700 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] svg { display: none; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #06B6D4 0%, #0891B2 100%);
        color: white; border: none; border-radius: 8px; 
        font-weight: 600; padding: 0.5rem 1.5rem;
        transition: all 0.2s; box-shadow: 0 2px 8px rgba(6, 182, 212, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #22D3EE 0%, #06B6D4 100%);
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.4);
        transform: translateY(-1px);
    }
    
    /* Dataframes */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 8px; padding: 8px 20px;
        color: #94A3B8; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(6, 182, 212, 0.12); color: #06B6D4;
        border-bottom: 2px solid #06B6D4;
    }
    
    /* File uploader */
    section[data-testid="stFileUploadDropzone"] {
        background: rgba(6, 182, 212, 0.05);
        border: 2px dashed rgba(6, 182, 212, 0.3);
        border-radius: 12px;
    }
    
    /* Success/Info boxes */
    .stAlert { border-radius: 10px; }
    
    /* Custom classes */
    .hero-title { 
        font-size: 2.4rem; font-weight: 700; 
        background: linear-gradient(135deg, #06B6D4, #8B5CF6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle { color: #64748B; font-size: 1.1rem; margin-top: 4px; }
    .card {
        background: linear-gradient(135deg, #1E293B 0%, #1A2332 100%);
        border: 1px solid rgba(6, 182, 212, 0.1);
        border-radius: 12px; padding: 24px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }
    .status-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-submitted { background: rgba(34, 197, 94, 0.15); color: #22C55E; }
    .badge-rejected { background: rgba(239, 68, 68, 0.15); color: #EF4444; }
    .badge-processing { background: rgba(6, 182, 212, 0.15); color: #06B6D4; }
    .node-flow {
        display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
        margin: 16px 0;
    }
    .node-box {
        background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 8px; padding: 8px 14px; font-size: 0.8rem;
        font-weight: 500; color: #CBD5E1;
    }
    .node-box.active { background: rgba(6, 182, 212, 0.2); color: #06B6D4; border-color: #06B6D4; }
    .node-arrow { color: #475569; font-size: 1.2rem; }
    .stat-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .stat-label { color: #94A3B8; font-size: 0.85rem; }
    .stat-value { color: #F1F5F9; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🎯 Tender Agent")
    st.caption("Autonomous AI Tender Pipeline")
    st.divider()
    
    # Agent status
    import os
    dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")
    if dry_run:
        st.warning("🔸 DRY RUN MODE", icon="⚡")
        st.caption("Using mock data. Set `DRY_RUN=false` for production.")
    else:
        st.success("🟢 LIVE MODE", icon="✅")
    
    st.divider()
    st.caption("Built with LangGraph + Claude")
    st.caption("© 2026 Acme SDS Solutions")

# ---------------------------------------------------------------------------
# Home Page Content
# ---------------------------------------------------------------------------

st.markdown('<p class="hero-title">Command Center</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-powered tender discovery, drafting, and submission — fully autonomous.</p>', unsafe_allow_html=True)
st.markdown("")

# --- Quick Stats ---
col1, col2, col3, col4, col5 = st.columns(5)

# Initialize session state for tracking runs
if "dashboard_runs" not in st.session_state:
    st.session_state.dashboard_runs = []

runs = st.session_state.dashboard_runs
submitted = sum(1 for r in runs if r.get("status") == "submitted")
rejected = sum(1 for r in runs if r.get("status") == "rejected")
total_sections = sum(len(r.get("drafted_sections", [])) for r in runs)
total_cost = sum(r.get("_cost", 0) for r in runs)

col1.metric("Tenders Processed", len(runs))
col2.metric("Submitted", submitted, delta=f"{submitted}" if submitted else None)
col3.metric("Rejected", rejected)
col4.metric("Sections Drafted", total_sections)
col5.metric("Est. Cost", f"${total_cost:.2f}")

st.markdown("")

# --- Pipeline Flow Diagram ---
st.markdown("### Pipeline Architecture")
st.markdown("""
<div class="card">
    <div class="node-flow">
        <div class="node-box active">🔍 Discover</div>
        <span class="node-arrow">→</span>
        <div class="node-box active">📊 Evaluate</div>
        <span class="node-arrow">→</span>
        <div class="node-box active">✍️ Draft</div>
        <span class="node-arrow">→</span>
        <div class="node-box active">🔎 Gap Check</div>
        <span class="node-arrow">→</span>
        <div class="node-box">💬 Slack</div>
        <span class="node-arrow">→</span>
        <div class="node-box active">📄 Assemble</div>
        <span class="node-arrow">→</span>
        <div class="node-box active">🚀 Submit</div>
    </div>
    <div style="color: #64748B; font-size: 0.8rem; margin-top: 8px;">
        7-node LangGraph state machine • Multi-model routing (Haiku → Sonnet → Opus) • PostgreSQL checkpointing
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# --- Quick Actions ---
st.markdown("### Quick Actions")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    <div class="card">
        <h4 style="color: #06B6D4; margin-top: 0;">🔍 Discover Tenders</h4>
        <p style="color: #94A3B8; font-size: 0.85rem;">
            Scan SAM.gov and email inbox for new opportunities matching your capabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Run Discovery", key="btn_discover", use_container_width=True):
        st.switch_page("pages/2_🔍_Discovery.py")

with col_b:
    st.markdown("""
    <div class="card">
        <h4 style="color: #8B5CF6; margin-top: 0;">📁 Knowledge Base</h4>
        <p style="color: #94A3B8; font-size: 0.85rem;">
            Upload company documents — profiles, certifications, past tenders, pricing.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Manage Documents", key="btn_kb", use_container_width=True):
        st.switch_page("pages/4_📁_Knowledge_Base.py")

with col_c:
    st.markdown("""
    <div class="card">
        <h4 style="color: #F59E0B; margin-top: 0;">📋 Pipeline</h4>
        <p style="color: #94A3B8; font-size: 0.85rem;">
            View all tenders in progress — scores, drafts, gaps, and submission status.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Pipeline", key="btn_pipeline", use_container_width=True):
        st.switch_page("pages/3_📋_Pipeline.py")

st.markdown("")

# --- Recent Activity ---
if runs:
    st.markdown("### Recent Activity")
    for run in reversed(runs[-5:]):
        status = run.get("status", "unknown")
        icon = "✅" if status == "submitted" else ("⛔" if status == "rejected" else "⏳")
        badge_class = "badge-submitted" if status == "submitted" else (
            "badge-rejected" if status == "rejected" else "badge-processing"
        )
        title = run.get("tender_title", "Untitled")[:60]
        score = run.get("eval_score", "?")
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; padding: 10px 16px; 
             background: rgba(30, 41, 59, 0.5); border-radius: 8px; margin-bottom: 6px;
             border-left: 3px solid {'#22C55E' if status == 'submitted' else ('#EF4444' if status == 'rejected' else '#06B6D4')};">
            <span style="font-size: 1.2rem;">{icon}</span>
            <div style="flex: 1;">
                <div style="color: #E2E8F0; font-weight: 500; font-size: 0.9rem;">{title}</div>
                <div style="color: #64748B; font-size: 0.75rem;">Score: {score}/100 • {run.get('source_portal', '?')}</div>
            </div>
            <span class="status-badge {badge_class}">{status}</span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No tenders processed yet. Click **Run Discovery** to get started!", icon="💡")