"""Dashboard page — Analytics, costs, and performance metrics."""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard — Tender Agent", page_icon="📊", layout="wide")

st.markdown("## 📊 Analytics Dashboard")
st.caption("Performance metrics, cost tracking, and pipeline analytics.")

st.divider()

runs = st.session_state.get("dashboard_runs", [])

if not runs:
    st.info("No data yet. Process some tenders from the **Discovery** page to see analytics here.", icon="📊")
    st.stop()

# --- Top Metrics ---
c1, c2, c3, c4, c5 = st.columns(5)

submitted = sum(1 for r in runs if r.get("status") == "submitted")
rejected = sum(1 for r in runs if r.get("status") == "rejected")
scores = [r.get("eval_score", 0) for r in runs if r.get("eval_score")]
avg_score = sum(scores) / len(scores) if scores else 0
total_sections = sum(len(r.get("drafted_sections", [])) for r in runs)
total_cost = sum(r.get("_cost", 0) for r in runs)

c1.metric("Win Rate", f"{submitted/len(runs)*100:.0f}%" if runs else "0%")
c2.metric("Avg Score", f"{avg_score:.0f}/100")
c3.metric("Tenders", len(runs))
c4.metric("Sections", total_sections)
c5.metric("Total Cost", f"${total_cost:.2f}")

st.markdown("")

# --- Charts ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Tender Outcomes")
    outcomes = {"Submitted ✅": submitted, "Rejected ⛔": rejected}
    other = len(runs) - submitted - rejected
    if other > 0:
        outcomes["Other"] = other
    df_out = pd.DataFrame({"Status": outcomes.keys(), "Count": outcomes.values()})
    st.bar_chart(df_out.set_index("Status"), color="#06B6D4")

with col_right:
    st.markdown("#### Evaluation Scores")
    if scores:
        score_data = pd.DataFrame({
            "Tender": [r.get("tender_id", "?")[:15] for r in runs if r.get("eval_score")],
            "Score": scores
        })
        st.bar_chart(score_data.set_index("Tender"), color="#8B5CF6")

st.markdown("")

# --- Model Usage ---
st.markdown("#### Model Routing Breakdown")
model_counts = {"Qwen3.5 Plus (Standard)": 0, "Qwen3 Max (Compliance)": 0, "Qwen3.5 Flash (Eval)": 0}
for r in runs:
    for s in r.get("drafted_sections", []):
        model = s.get("model_used", "")
        if model == "qwen3-max":
            model_counts["Qwen3 Max (Compliance)"] += 1
        elif model == "qwen3.5-plus":
            model_counts["Qwen3.5 Plus (Standard)"] += 1
    for e in r.get("audit_log", []):
        model = e.get("model_used") or ""
        if "qwen3.5-flash" in model:
            model_counts["Qwen3.5 Flash (Eval)"] += 1

mc1, mc2, mc3 = st.columns(3)
mc1.metric("Qwen3.5 Flash Calls", model_counts["Qwen3.5 Flash (Eval)"], help="Fast eligibility scoring")
mc2.metric("Qwen3.5 Plus Calls", model_counts["Qwen3.5 Plus (Standard)"], help="Section drafting & gap checking")
mc3.metric("Qwen3 Max Calls", model_counts["Qwen3 Max (Compliance)"], help="Compliance-critical sections")

st.markdown("")

# --- Source Distribution ---
st.markdown("#### Discovery Sources")
sources = {}
for r in runs:
    src = r.get("source_portal", "unknown")
    sources[src] = sources.get(src, 0) + 1

src_df = pd.DataFrame({"Source": sources.keys(), "Tenders": sources.values()})
st.bar_chart(src_df.set_index("Source"), color="#F59E0B")

st.markdown("")

# --- Budget Monitor ---
st.markdown("#### Budget Monitor")
budget = 300.0
spent = total_cost
remaining = budget - spent
pct = (spent / budget * 100) if budget else 0

b1, b2, b3 = st.columns(3)
b1.metric("Budget", f"${budget:.0f}/month")
b2.metric("Spent", f"${spent:.2f}")
b3.metric("Remaining", f"${remaining:.2f}")

st.progress(min(pct / 100, 1.0), text=f"{pct:.1f}% of monthly budget used")

if pct >= 100:
    st.error("🔴 Budget exceeded! Consider reducing tender volume or switching to cheaper models.")
elif pct >= 80:
    st.warning("⚠️ Approaching budget limit. Monitor closely.")
else:
    st.success("✅ Spending is within budget.")