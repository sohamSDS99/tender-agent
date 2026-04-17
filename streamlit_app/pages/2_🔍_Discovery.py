"""Discovery page — Find and evaluate new tenders."""
import streamlit as st
import sys, os, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DRY_RUN", "true")

st.set_page_config(page_title="Discovery — Tender Agent", page_icon="🔍", layout="wide")

st.markdown("## 🔍 Tender Discovery")
st.caption("Scan SAM.gov and email inbox for new procurement opportunities.")

st.divider()

# Controls
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    days_back = st.slider("Days to search", 1, 30, 7)
with col2:
    min_relevance = st.slider("Min relevance score", 0.0, 1.0, 0.15, 0.05)
with col3:
    st.markdown("")
    run_discovery = st.button("🔍 Scan Now", type="primary", use_container_width=True)

if run_discovery:
    with st.spinner("Scanning SAM.gov and email inbox..."):
        from src.discovery.coordinator import DiscoveryCoordinator
        coordinator = DiscoveryCoordinator(min_relevance=min_relevance)
        tenders = coordinator.discover_new_tenders(days_back=days_back)
        st.session_state["discovered_tenders"] = tenders
        time.sleep(0.5)  # Visual feedback

if "discovered_tenders" in st.session_state:
    tenders = st.session_state["discovered_tenders"]
    
    st.success(f"Found **{len(tenders)}** relevant tenders", icon="✅")
    st.markdown("")
    
    # Results table
    for i, t in enumerate(tenders):
        source = t["source_portal"]
        source_icon = "🏛️" if source == "sam.gov" else "📧"
        title = t["tender_title"]
        deadline = t.get("submission_deadline", "N/A")[:10]
        
        with st.expander(f"{source_icon} **{title}**", expanded=(i == 0)):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown(f"**Source:** {source}")
                st.markdown(f"**ID:** `{t['tender_id']}`")
                st.markdown(f"**Deadline:** {deadline}")
            with c2:
                st.markdown(f"**URL:** [Link]({t.get('source_url', '#')})")
            with c3:
                if st.button(f"▶️ Process", key=f"process_{i}", use_container_width=True):
                    st.session_state["tender_to_process"] = t
                    st.session_state["processing_index"] = i
            
            st.markdown("---")
            st.markdown("**Description:**")
            st.markdown(f"<div style='color: #94A3B8; font-size: 0.85rem;'>{t['tender_raw_text'][:500]}...</div>", unsafe_allow_html=True)
    
    # Process selected tender
    if "tender_to_process" in st.session_state:
        tender = st.session_state.pop("tender_to_process")
        idx = st.session_state.pop("processing_index", 0)
        
        st.markdown("---")
        st.markdown(f"### ⚙️ Processing: {tender['tender_title'][:60]}")
        
        progress = st.progress(0, text="Starting pipeline...")
        
        import src.agent.graph as gm
        from src.agent.nodes.discover import discover_node
        from src.agent.nodes.evaluate import evaluate_node
        from src.agent.nodes.retrieve_draft import retrieve_draft_node
        from src.agent.nodes.gap_check import gap_check_node
        from src.agent.nodes.slack_escalate import slack_escalate_node
        from src.agent.nodes.assemble import assemble_node
        from src.agent.nodes.submit import submit_node
        
        orig = {}
        for name, fn in [("discover", discover_node), ("evaluate", evaluate_node),
                          ("retrieve_draft", retrieve_draft_node), ("gap_check", gap_check_node),
                          ("slack_escalate", slack_escalate_node), ("assemble", assemble_node),
                          ("submit", submit_node)]:
            orig[name] = getattr(gm, f"{name}_node")
            setattr(gm, f"{name}_node", fn)
        
        try:
            from src.agent.graph import build_tender_graph
            graph = build_tender_graph(checkpointer=None)
            
            progress.progress(15, text="Discovering...")
            time.sleep(0.3)
            progress.progress(30, text="Evaluating eligibility...")
            time.sleep(0.3)
            progress.progress(50, text="Drafting response sections...")
            time.sleep(0.3)
            progress.progress(70, text="Running gap check...")
            time.sleep(0.3)
            
            result = graph.invoke(tender)
            
            progress.progress(90, text="Submitting...")
            time.sleep(0.3)
            progress.progress(100, text="Complete!")
            
            # Save to session
            if "dashboard_runs" not in st.session_state:
                st.session_state.dashboard_runs = []
            result["_cost"] = 0.037  # Estimated dry-run cost
            st.session_state.dashboard_runs.append(result)
            
            status = result["status"]
            if status == "submitted":
                st.success(f"✅ Tender submitted! Confirmation: **{result.get('submission_confirmation', 'N/A')}**")
            elif status == "rejected":
                st.warning(f"⛔ Tender rejected (Score: {result.get('eval_score', '?')}/100)")
            else:
                st.error(f"Status: {status}")
            
            # Show details
            with st.expander("📊 Processing Details", expanded=True):
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Eval Score", f"{result.get('eval_score', '?')}/100")
                d2.metric("Sections", len(result.get("drafted_sections", [])))
                d3.metric("Gaps Found", len(result.get("gaps", [])))
                d4.metric("Method", result.get("submission_method", "N/A"))
                
                if result.get("audit_log"):
                    st.markdown("**Audit Trail:**")
                    for entry in result["audit_log"]:
                        st.markdown(f"- `{entry['node']}` → {entry['action']}")
        finally:
            for name, fn in orig.items():
                setattr(gm, f"{name}_node", fn)
else:
    st.info("Click **Scan Now** to discover new tender opportunities.", icon="🔍")