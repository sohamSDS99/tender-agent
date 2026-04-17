"""Knowledge Base page — Upload and manage company documents."""
import streamlit as st
import sys, os, tempfile
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DRY_RUN", "true")

st.set_page_config(page_title="Knowledge Base — Tender Agent", page_icon="📁", layout="wide")

st.markdown("## 📁 Knowledge Base")
st.caption("Upload company documents that the agent uses to draft tender responses.")

st.divider()

# Init session state for KB tracking
if "kb_documents" not in st.session_state:
    st.session_state.kb_documents = []
if "kb_chunks" not in st.session_state:
    st.session_state.kb_chunks = []

# --- Upload Section ---
st.markdown("### Upload Documents")
st.markdown("""
<div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 12px;">
    Upload company profiles, certifications, past tender responses, pricing documents, 
    and team bios. Supported formats: <b>PDF, DOCX, TXT, MD</b>.
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop files here",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    if st.button("📥 Ingest Documents", type="primary"):
        from src.ingestion.parser import DocumentParser
        from src.ingestion.chunker import TextChunker
        
        parser = DocumentParser()
        chunker = TextChunker(chunk_size=2000, chunk_overlap=200)
        
        progress = st.progress(0, text="Starting ingestion...")
        total = len(uploaded_files)
        
        all_chunks = []
        
        for i, file in enumerate(uploaded_files):
            progress.progress((i + 1) / total, text=f"Processing {file.name}...")
            
            # Save to temp file
            suffix = Path(file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            try:
                pages = parser.parse(tmp_path)
                chunks = chunker.chunk_pages(pages)
                
                # Fix source_file to use original name
                for c in chunks:
                    c.source_file = file.name
                
                all_chunks.extend(chunks)
                
                st.session_state.kb_documents.append({
                    "name": file.name,
                    "pages": len(pages),
                    "chunks": len(chunks),
                    "chars": sum(c.char_count for c in chunks),
                    "size_kb": file.size / 1024,
                })
            except Exception as exc:
                st.error(f"Failed to process {file.name}: {exc}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        
        st.session_state.kb_chunks.extend(all_chunks)
        progress.progress(1.0, text="Complete!")
        st.success(f"✅ Ingested **{len(uploaded_files)}** documents → **{len(all_chunks)}** chunks")

st.markdown("")

# --- Current Knowledge Base ---
st.markdown("### Knowledge Base Contents")

docs = st.session_state.kb_documents
if docs:
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(docs))
    c2.metric("Total Chunks", sum(d["chunks"] for d in docs))
    c3.metric("Total Characters", f"{sum(d['chars'] for d in docs):,}")
    
    st.markdown("")
    
    import pandas as pd
    df = pd.DataFrame(docs)
    df.columns = ["Filename", "Pages", "Chunks", "Characters", "Size (KB)"]
    df["Size (KB)"] = df["Size (KB)"].round(1)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("")
    
    # Preview chunks
    chunks = st.session_state.kb_chunks
    if chunks:
        st.markdown("### Chunk Preview")
        st.caption(f"Showing first 10 of {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:10]):
            with st.expander(
                f"Chunk {i+1} — **{chunk.source_file}** (page {chunk.page_number}) — {chunk.char_count} chars"
            ):
                heading = chunk.section_heading
                if heading:
                    st.markdown(f"**Section:** {heading}")
                st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
    
    # Clear button
    st.markdown("")
    if st.button("🗑️ Clear Knowledge Base", type="secondary"):
        st.session_state.kb_documents = []
        st.session_state.kb_chunks = []
        st.rerun()
else:
    st.info(
        "No documents uploaded yet. Upload company profiles, certifications, "
        "past tenders, and pricing documents to power the RAG pipeline.",
        icon="📁"
    )
    
    st.markdown("""
    <div style="margin-top: 16px; padding: 16px; background: rgba(6, 182, 212, 0.05); 
         border-radius: 10px; border: 1px solid rgba(6, 182, 212, 0.1);">
        <h4 style="color: #06B6D4; margin-top: 0;">💡 Recommended Documents</h4>
        <ul style="color: #94A3B8; font-size: 0.85rem;">
            <li><b>Company Profile</b> — Overview, history, client count, industries served</li>
            <li><b>Certifications</b> — ISO 27001, SOC 2, FedRAMP, etc.</li>
            <li><b>Capability Statements</b> — Platform features, GHS, OSHA compliance</li>
            <li><b>Past Tender Responses</b> — Winning proposals for reference</li>
            <li><b>Pricing Templates</b> — Standard pricing tiers</li>
            <li><b>Team Bios</b> — Key personnel, qualifications</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)