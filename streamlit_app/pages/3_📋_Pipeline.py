"""Pipeline page — View all processed tenders and their status."""
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Pipeline — Tender Agent", page_icon="📋", layout="wide")

st.markdown("## 📋 Tender Pipeline")
st.caption("Track all tenders through the 7-stage processing pipeline.")

st.divider()

runs = st.session_state.get("dashboard_runs", [])

if not runs:
    st.info("No tenders processed yet. Go to **Discovery** to find and process tenders.", icon="📋")
    st.stop()

# --- Summary metrics ---
c1, c2, c3, c4 = st.columns(4)
submitted = sum(1 for r in runs if r.get("status") == "submitted")
rejected = sum(1 for r in runs if r.get("status") == "rejected")
avg_score = sum(r.get("eval_score", 0) for r in runs) / len(runs)
c1.metric("Total", len(runs))
c2.metric("Submitted", submitted)
c3.metric("Rejected", rejected)
c4.metric("Avg Score", f"{avg_score:.0f}/100")

st.markdown("")

# --- Tender table ---
data = []
for r in runs:
    data.append({
        "ID": r.get("tender_id", "?")[:20],
        "Title": r.get("tender_title", "?")[:45],
        "Source": r.get("source_portal", "?"),
        "Score": r.get("eval_score", 0),
        "Decision": r.get("eval_decision", "?"),
        "Sections": len(r.get("drafted_sections", [])),
        "Gaps": len(r.get("gaps", [])),
        "Status": r.get("status", "?"),
        "Method": r.get("submission_method", "N/A"),
        "Confirmation": r.get("submission_confirmation", "N/A")[:15],
    })

df = pd.DataFrame(data)

# Color-code the status column
def highlight_status(val):
    if val == "submitted":
        return "color: #22C55E; font-weight: 600;"
    elif val == "rejected":
        return "color: #EF4444; font-weight: 600;"
    return "color: #06B6D4;"

styled = df.style.applymap(highlight_status, subset=["Status"])
st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

st.markdown("")

# --- Tender detail view ---
st.markdown("### Tender Detail View")
selected_idx = st.selectbox(
    "Select a tender to inspect",
    range(len(runs)),
    format_func=lambda i: f"{runs[i].get('tender_id', '?')} — {runs[i].get('tender_title', '?')[:50]}",
)

if selected_idx is not None:
    r = runs[selected_idx]
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Evaluation", "✍️ Drafted Sections", "📄 Document", "📜 Audit"])
    
    with tab1:
        st.markdown(f"**Score:** {r.get('eval_score', '?')}/100 — **Decision:** {r.get('eval_decision', '?').upper()}")
        st.markdown(f"**Reasoning:** {r.get('eval_reasoning', 'N/A')}")
        
        breakdown = r.get("eval_breakdown", {})
        if breakdown:
            bd_df = pd.DataFrame([
                {"Dimension": k.replace("_", " ").title(), "Score": v}
                for k, v in breakdown.items()
            ])
            st.bar_chart(bd_df.set_index("Dimension"))
    
    with tab2:
        sections = r.get("drafted_sections", [])
        if sections:
            for s in sections:
                model_tag = "🟣 Qwen3 Max" if s.get("model_used", "") == "qwen3-max" else "🔵 Qwen3.5 Plus"
                conf = s.get("confidence", 0)
                conf_color = "#22C55E" if conf >= 0.8 else ("#F59E0B" if conf >= 0.6 else "#EF4444")
                
                with st.expander(f"**{s.get('section_id', '')} {s.get('section_title', '')}** — {model_tag} — Confidence: {conf:.0%}"):
                    st.markdown(s.get("content", "No content"))
        else:
            st.info("No sections drafted (tender may have been rejected)")
    
    with tab3:
        doc_path = r.get("assembled_document_path", "")
        if doc_path and Path(doc_path).exists():
            content = Path(doc_path).read_text(encoding="utf-8")
            st.markdown(f"**File:** `{Path(doc_path).name}` — {len(content.split())} words")
            st.divider()
            st.markdown(content)
            st.download_button("⬇️ Download Document", content, Path(doc_path).name, mime="text/markdown")
        else:
            st.info("No assembled document available")
    
    with tab4:
        audit = r.get("audit_log", [])
        if audit:
            for j, e in enumerate(audit, 1):
                ts = e.get("timestamp", "")[:19].replace("T", " ")
                tokens = e.get("tokens_used", 0)
                tok_str = f" • `{tokens} tokens`" if tokens else ""
                st.markdown(
                    f"**{j}.** `{ts}` — **{e.get('node', '?')}** → {e.get('action', '?')}{tok_str}"
                )
                if e.get("detail"):
                    st.caption(e["detail"][:200])
        else:
            st.info("No audit trail available")