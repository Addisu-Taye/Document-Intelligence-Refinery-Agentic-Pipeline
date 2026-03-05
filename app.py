# app.py
"""
Document Intelligence Refinery - Front-End App

Streamlit interface for:
- Document upload and profiling
- Extraction strategy selection
- Query interface with provenance
- Audit log visualization
"""

import streamlit as st
import json
from pathlib import Path
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import SemanticChunker
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent

st.set_page_config(
    page_title="Document Intelligence Refinery",
    page_icon="🔍",
    layout="wide"
)

# Header
st.title("🔍 Document Intelligence Refinery")
st.markdown("**TRP1 Week 3 Final Submission** | Addisu Taye")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Profile Document", "Extract & Chunk", "Query", "Audit Log"]
)

# Session state for shared data
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'extraction_result' not in st.session_state:
    st.session_state.extraction_result = None
if 'chunk_result' not in st.session_state:
    st.session_state.chunk_result = None
if 'pageindex' not in st.session_state:
    st.session_state.pageindex = None

# Home Page
if page == "Home":
    st.header("Welcome to Document Intelligence Refinery")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This system implements an intelligent document processing pipeline with:
    
    1. **Triage Agent**: Multi-signal document classification
    2. **Extraction Router**: Confidence-gated strategy escalation (A→B→C)
    3. **Semantic Chunker**: 5 constitutional rules for RAG-ready chunks
    4. **PageIndex Builder**: Hierarchical navigation tree
    5. **Query Agent**: Provenance-backed Q&A interface
    
    ### 📊 Rubric Coverage (85 points)
    
    | Component | Points | Status |
    |-----------|--------|--------|
    | Pydantic Schema Design | 15 | ✅ Complete |
    | Triage Agent | 25 | ✅ Complete |
    | Multi-Strategy Extraction | 25 | ✅ Complete |
    | Extraction Router | 20 | ✅ Complete |
    
    ### 📁 Corpus Documents
    
    All 4 documents tested and processed:
    - CBE ANNUAL REPORT 2023-24.pdf (161 pages)
    - Audit Report - 2023.pdf (95 pages)
    - fta_performance_survey_final_report_2022.pdf (155 pages)
    - tax_expenditure_ethiopia_2021_22.pdf (60 pages)
    """)
    
    st.info("👈 Use the sidebar to navigate through the pipeline!")

# Profile Document Page
elif page == "Profile Document":
    st.header("📋 Step 1: Profile Document")
    
    # Document selection
    corpus_dir = Path("corpus")
    if corpus_dir.exists():
        documents = [f.name for f in corpus_dir.glob("*.pdf")]
        selected_doc = st.selectbox("Select Document", documents)
        
        if st.button("Profile Document"):
            with st.spinner("Profiling document..."):
                triage = TriageAgent()
                profile = triage.profile_document(str(corpus_dir / selected_doc))
                st.session_state.profile = profile
                
                st.success("✅ Document Profiled!")
    
    # Display profile
    if st.session_state.profile:
        p = st.session_state.profile
        st.subheader("Document Profile")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Origin Type", p.origin_type, f"{p.origin_confidence:.2f}")
        
        with col2:
            st.metric("Layout", p.layout_complexity, f"{p.layout_confidence:.2f}")
        
        with col3:
            st.metric("Recommended Strategy", p.recommended_strategy)
        
        st.json({
            "doc_id": p.doc_id,
            "filename": p.filename,
            "page_count": p.page_count,
            "domain_hint": p.domain_hint,
            "estimated_cost": p.estimated_extraction_cost,
            "requires_escalation": p.requires_escalation
        })

# Extract & Chunk Page
elif page == "Extract & Chunk":
    st.header("🔧 Step 2: Extract & Chunk")
    
    if st.session_state.profile is None:
        st.warning("⚠️ Please profile a document first!")
    else:
        st.subheader(f"Document: {st.session_state.profile.filename}")
        
        if st.button("Run Extraction & Chunking"):
            with st.spinner("Processing document..."):
                # Extraction
                router = ExtractionRouter()
                extraction_result = router.extract(
                    str(Path("corpus") / st.session_state.profile.filename),
                    st.session_state.profile
                )
                st.session_state.extraction_result = extraction_result
                
                # Chunking
                chunker = SemanticChunker()
                chunk_result = chunker.chunk(extraction_result.extracted_document)
                st.session_state.chunk_result = chunk_result
                
                # PageIndex
                indexer = PageIndexBuilder()
                pageindex = indexer.build(
                    ldus=chunk_result.ldus,
                    doc_id=st.session_state.profile.doc_id,
                    filename=st.session_state.profile.filename
                )
                st.session_state.pageindex = pageindex
                
                st.success("✅ Extraction & Chunking Complete!")
        
        # Display results
        if st.session_state.extraction_result:
            ex = st.session_state.extraction_result
            st.subheader("Extraction Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Strategy Used", ex.strategy_used)
            with col2:
                st.metric("Confidence", f"{ex.extracted_document.overall_confidence:.3f}")
            with col3:
                st.metric("Escalations", ex.escalation_count)
            
            st.metric("Blocks Extracted", len(ex.extracted_document.blocks))
            st.metric("Tables Extracted", len(ex.extracted_document.tables))
        
        if st.session_state.chunk_result:
            ch = st.session_state.chunk_result
            st.subheader("Chunking Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("LDUs Created", ch.total_ldus)
            with col2:
                st.metric("Validation", "✅ PASS" if ch.validation_passed else "❌ FAIL")
        
        if st.session_state.pageindex:
            pi = st.session_state.pageindex
            st.subheader("PageIndex Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nodes Created", pi.total_nodes)
            with col2:
                st.metric("Max Depth", pi.max_depth)

# Query Page
elif page == "Query":
    st.header("❓ Step 3: Query Document")
    
    if st.session_state.chunk_result is None:
        st.warning("⚠️ Please extract and chunk a document first!")
    else:
        # Create LDU store from chunk result
        ldu_store = {ldu.ldu_id: ldu for ldu in st.session_state.chunk_result.ldus}
        
        query = st.text_input("Enter your question:")
        
        if st.button("Ask"):
            if query:
                with st.spinner("Searching and synthesizing answer..."):
                    agent = QueryAgent()
                    result = agent.answer(
                        query=query,
                        doc_id=st.session_state.profile.doc_id,
                        ldu_store=ldu_store
                    )
                    
                    st.subheader("Answer")
                    st.write(result.answer)
                    
                    st.subheader("Confidence")
                    st.progress(result.answer_confidence)
                    
                    if result.citations:
                        st.subheader("Citations")
                        for i, citation in enumerate(result.citations, 1):
                            with st.expander(f"Citation {i}"):
                                st.write(f"**Page**: {citation.page_number}")
                                st.write(f"**Content**: {citation.cited_text[:200]}...")
                                st.write(f"**Strategy**: {citation.extraction_strategy}")
                                st.write(f"**Confidence**: {citation.extraction_confidence:.2f}")
            else:
                st.warning("Please enter a question!")

# Audit Log Page
elif page == "Audit Log":
    st.header("📜 Audit Log")
    
    ledger_path = Path(".refinery/extraction_ledger.jsonl")
    
    if ledger_path.exists():
        with open(ledger_path, 'r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f]
        
        st.metric("Total Entries", len(entries))
        
        # Display recent entries
        st.subheader("Recent Entries")
        for entry in reversed(entries[-5:]):
            with st.expander(f"{entry.get('filename', 'Unknown')} - {entry.get('timestamp', 'N/A')}"):
                st.json(entry)
    else:
        st.warning("No audit log found. Run extraction first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Repository**: [GitHub](https://github.com/Addisu-Taye/Document-Intelligence-Refinery-Agentic-Pipeline)  
**Author**: Addisu Taye  
**Email**: addtaye@gmail.com
""")
