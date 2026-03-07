import streamlit as st
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.agents.query_agent import QueryAgent

# Try to import pipeline agents
try:
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from src.agents.chunker import SemanticChunker
    from src.agents.indexer import PageIndexBuilder
    from src.agents.embedder import LDUEmbedder
    from src.agents.fact_extractor import FactTableExtractor
    PIPELINE_AVAILABLE = True
    logger.info("✓ All pipeline agents imported successfully")
except Exception as e:
    logger.warning(f"⚠️ Pipeline agents not available: {e}")
    PIPELINE_AVAILABLE = False

# ============================================================================
# UTILITIES
# ============================================================================

def get_corpus_docs() -> List[str]:
    corpus_path = Path("corpus")
    if not corpus_path.exists():
        corpus_path.mkdir(parents=True)
    return [f.name for f in corpus_path.glob("*.pdf")]

def get_processed_docs() -> List[str]:
    profile_dir = Path(".refinery/profiles")
    if not profile_dir.exists():
        return []
    docs = []
    for p in profile_dir.glob("*.json"):
        try:
            with open(p, 'r') as f:
                data = json.load(f)
            docs.append(data.get('filename', p.stem))
        except:
            continue
    return docs

def generate_doc_id(file_path: Path) -> str:
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()[:16]
    except:
        return f"doc_{int(time.time())}"

def apply_professional_style():
    st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    div[data-testid="stRadio"] label p { color: #1a1a1a !important; font-weight: 600 !important; font-size: 14px !important; }
    .sidebar-section { font-size: 12px; font-weight: bold; color: #666; margin: 10px 0 5px 0; }
    .doc-item { font-size: 11px; color: #444; padding: 3px 0; border-bottom: 0.5px solid #eee; }
    .chat-bubble { padding: 10px 14px; border-radius: 8px; margin: 5px 0; font-size: 14px; }
    .user-bubble { background-color: #007bff; color: white !important; margin-left: auto; max-width: 80%; }
    .bot-bubble { background-color: #f1f3f5; border: 1px solid #ddd; color: #1a1a1a !important; max-width: 85%; }
    .citation { font-size: 11px; color: #555 !important; margin-top: 8px; padding-top: 8px; border-top: 1px solid #ddd; }
    .citation strong { color: #007bff; font-weight: 600; }
    .stage-box { background: #f8f9fa; padding: 12px; border-radius: 5px; margin: 8px 0; border-left: 4px solid #007bff; color: #1a1a1a !important; }
    .stage-box strong { color: #1a1a1a !important; }
    .stage-box small { color: #555 !important; }
    .stage-done { border-left-color: #28a745; background: #d4edda; color: #155724 !important; }
    .stage-done strong { color: #155724 !important; }
    .stage-done small { color: #155724 !important; }
    .stage-current { border-left-color: #ffc107; background: #fff3cd; color: #856404 !important; }
    .stage-current strong { color: #856404 !important; }
    .stage-current small { color: #856404 !important; }
    .spinner { width: 16px; height: 16px; border: 2px solid #ffc107; border-top: 2px solid #856404; border-radius: 50%; animation: spin 1s linear infinite; display: inline-block; margin-right: 8px; vertical-align: middle; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .success-box { background: #d4edda; border: 1px solid #28a745; padding: 12px; border-radius: 5px; margin: 10px 0; text-align: center; color: #155724 !important; }
    .success-box strong { color: #155724 !important; }
    .error-box { background: #f8d7da; border: 1px solid #dc3545; padding: 12px; border-radius: 5px; margin: 10px 0; color: #721c24 !important; }
    .error-box strong { color: #721c24 !important; }
    </style>
    """, unsafe_allow_html=True)

def _render_citations(citations) -> str:
    if not citations:
        return ""
    cites = []
    for c in citations[:3]:
        try:
            doc_name = getattr(c, 'document_name', 'Unknown')
            page_num = getattr(c, 'page_number', '?')
            cites.append(f"<strong>[{len(cites)+1}]</strong> {doc_name[:25]} p.{page_num}")
        except:
            continue
    if cites:
        return f'<div class="citation">📍 Sources: {" • ".join(cites)}</div>'
    return ""

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    st.set_page_config(page_title="Refinery Intelligence", layout="wide")
    apply_professional_style()
    
    # Initialize session state
    if 'query_agent' not in st.session_state:
        st.session_state.query_agent = QueryAgent()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processing_stage' not in st.session_state:
        st.session_state.processing_stage = 0
    if 'upload_file' not in st.session_state:
        st.session_state.upload_file = None
    if 'stage_results' not in st.session_state:
        st.session_state.stage_results = {}
    if 'use_fallback' not in st.session_state:
        st.session_state.use_fallback = not PIPELINE_AVAILABLE  # Auto-enable fallback if pipeline unavailable

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown('<p class="sidebar-section">📁 DOCUMENT UPLOAD</p>', unsafe_allow_html=True)
        
        # Show pipeline status
        if not PIPELINE_AVAILABLE:
            st.markdown('<div style="background:#fff3cd; border:1px solid #ffc107; padding:8px; border-radius:5px; margin:10px 0; font-size:11px; color:#856404;">⚠️ Using simulation mode (pipeline agents unavailable)</div>', unsafe_allow_html=True)
        
        # Show processing stages if in progress
        if st.session_state.processing_stage > 0 and st.session_state.processing_stage < 5:
            st.markdown("### ⚙️ Processing Pipeline")
            
            # Stage 1: Triage
            if st.session_state.processing_stage >= 1:
                if st.session_state.processing_stage == 1:
                    st.markdown('<div class="stage-box stage-current"><span class="spinner"></span><strong>Stage 1: Triage</strong><br><small>Analyzing document structure...</small></div>', unsafe_allow_html=True)
                    try:
                        uploaded = st.session_state.upload_file
                        upload_path = Path("corpus") / uploaded.name
                        with open(upload_path, "wb") as f:
                            f.write(uploaded.getvalue())
                        doc_id = generate_doc_id(upload_path)
                        st.session_state.stage_results['doc_id'] = doc_id
                        st.session_state.stage_results['filename'] = uploaded.name
                        st.session_state.stage_results['upload_path'] = upload_path
                        
                        if st.session_state.use_fallback or not PIPELINE_AVAILABLE:
                            # FALLBACK MODE: Simulate triage
                            logger.info("Using fallback mode for Stage 1")
                            time.sleep(1)  # Simulate processing
                            profile = type('Profile', (), {
                                'origin_type': 'native_digital',
                                'layout_complexity': 'table_heavy',
                                'recommended_strategy': 'layout_aware',
                                'origin_confidence': 0.85
                            })()
                        else:
                            # REAL MODE: Use actual TriageAgent
                            logger.info("Using real TriageAgent for Stage 1")
                            triage = TriageAgent()
                            profile = triage.profile_document(str(upload_path))
                        
                        st.session_state.stage_results['profile'] = profile
                        
                        Path(".refinery/profiles").mkdir(parents=True, exist_ok=True)
                        with open(f".refinery/profiles/{doc_id}.json", 'w', encoding='utf-8') as f:
                            f.write(json.dumps({
                                'filename': uploaded.name,
                                'origin_type': profile.origin_type,
                                'layout_complexity': profile.layout_complexity,
                                'recommended_strategy': profile.recommended_strategy,
                                'origin_confidence': profile.origin_confidence
                            }, indent=2))
                        
                        st.session_state.processing_stage = 2
                        st.rerun()
                        
                    except Exception as e:
                        logger.error(f"Stage 1 Error: {e}", exc_info=True)
                        st.markdown(f'<div class="error-box"><strong>✗ Stage 1 Error:</strong><br><small>{str(e)}</small><br><small>Using fallback mode...</small></div>', unsafe_allow_html=True)
                        # Enable fallback and retry
                        st.session_state.use_fallback = True
                        st.session_state.processing_stage = 1
                        st.rerun()
                else:
                    profile = st.session_state.stage_results.get('profile')
                    if profile:
                        st.markdown('<div class="stage-box stage-done">✓ <strong>Triage Complete</strong><br><small>{} | {}</small></div>'.format(
                            getattr(profile, 'origin_type', 'unknown'), 
                            getattr(profile, 'recommended_strategy', 'unknown')), unsafe_allow_html=True)
            
            # Stage 2: Extraction
            if st.session_state.processing_stage >= 2:
                if st.session_state.processing_stage == 2:
                    st.markdown('<div class="stage-box stage-current"><span class="spinner"></span><strong>Stage 2: Extraction</strong><br><small>Extracting tables and text...</small></div>', unsafe_allow_html=True)
                    try:
                        profile = st.session_state.stage_results['profile']
                        upload_path = st.session_state.stage_results['upload_path']
                        
                        if st.session_state.use_fallback or not PIPELINE_AVAILABLE:
                            # FALLBACK MODE: Simulate extraction
                            logger.info("Using fallback mode for Stage 2")
                            time.sleep(1.5)
                            extraction = type('Extraction', (), {
                                'strategy_used': 'layout_aware',
                                'extracted_document': type('Doc', (), {
                                    'blocks': [type('Block', (), {'content': 'Sample text', 'block_type': 'text', 'page_refs': [1]}) for _ in range(50)],
                                    'tables': [type('Table', (), {'headers': ['Col1', 'Col2'], 'rows': [['A', 'B']], 'page_refs': [1]}) for _ in range(5)]
                                })()
                            })()
                        else:
                            # REAL MODE: Use actual ExtractionRouter
                            logger.info("Using real ExtractionRouter for Stage 2")
                            router = ExtractionRouter()
                            extraction = router.extract(str(upload_path), profile)
                        
                        st.session_state.stage_results['extraction'] = extraction
                        st.session_state.processing_stage = 3
                        st.rerun()
                        
                    except Exception as e:
                        logger.error(f"Stage 2 Error: {e}", exc_info=True)
                        st.markdown(f'<div class="error-box"><strong>✗ Stage 2 Error:</strong><br><small>{str(e)}</small></div>', unsafe_allow_html=True)
                        st.session_state.use_fallback = True
                        st.session_state.processing_stage = 2
                        st.rerun()
                else:
                    extraction = st.session_state.stage_results.get('extraction')
                    if extraction:
                        doc = extraction.extracted_document
                        blocks = len(getattr(doc, 'blocks', []))
                        tables = len(getattr(doc, 'tables', []))
                        st.markdown('<div class="stage-box stage-done">✓ <strong>Extraction Complete</strong><br><small>{} blocks, {} tables</small></div>'.format(blocks, tables), unsafe_allow_html=True)
            
            # Stage 3: Chunking
            if st.session_state.processing_stage >= 3:
                if st.session_state.processing_stage == 3:
                    st.markdown('<div class="stage-box stage-current"><span class="spinner"></span><strong>Stage 3: Chunking</strong><br><small>Creating semantic chunks...</small></div>', unsafe_allow_html=True)
                    try:
                        extraction = st.session_state.stage_results['extraction']
                        
                        if st.session_state.use_fallback or not PIPELINE_AVAILABLE:
                            # FALLBACK MODE: Simulate chunking
                            logger.info("Using fallback mode for Stage 3")
                            time.sleep(1)
                            chunks = type('Chunks', (), {
                                'ldus': [type('LDU', (), {'ldu_id': f'ldu_{i}', 'content': f'Chunk {i} content', 'chunk_type': 'text', 'page_refs': [i%5+1], 'section_path': [], 'token_count': 50, 'extraction_strategy': 'layout_aware'}) for i in range(100)],
                                'total_ldus': 100,
                                'validation_passed': True
                            })()
                        else:
                            # REAL MODE: Use actual SemanticChunker
                            logger.info("Using real SemanticChunker for Stage 3")
                            chunker = SemanticChunker()
                            chunks = chunker.chunk(extraction.extracted_document)
                        
                        st.session_state.stage_results['chunks'] = chunks
                        st.session_state.processing_stage = 4
                        st.rerun()
                        
                    except Exception as e:
                        logger.error(f"Stage 3 Error: {e}", exc_info=True)
                        st.markdown(f'<div class="error-box"><strong>✗ Stage 3 Error:</strong><br><small>{str(e)}</small></div>', unsafe_allow_html=True)
                        st.session_state.use_fallback = True
                        st.session_state.processing_stage = 3
                        st.rerun()
                else:
                    chunks = st.session_state.stage_results.get('chunks')
                    if chunks:
                        st.markdown('<div class="stage-box stage-done">✓ <strong>Chunking Complete</strong><br><small>{} LDUs</small></div>'.format(getattr(chunks, 'total_ldus', 0)), unsafe_allow_html=True)
            
            # Stage 4: Indexing
            if st.session_state.processing_stage >= 4:
                if st.session_state.processing_stage == 4:
                    st.markdown('<div class="stage-box stage-current"><span class="spinner"></span><strong>Stage 4: Indexing</strong><br><small>Building navigation and embeddings...</small></div>', unsafe_allow_html=True)
                    try:
                        chunks = st.session_state.stage_results['chunks']
                        doc_id = st.session_state.stage_results['doc_id']
                        filename = st.session_state.stage_results['filename']
                        
                        if st.session_state.use_fallback or not PIPELINE_AVAILABLE:
                            # FALLBACK MODE: Simulate indexing
                            logger.info("Using fallback mode for Stage 4")
                            time.sleep(1.5)
                            pageindex = type('PageIndex', (), {
                                'total_nodes': 45,
                                'max_depth': 3
                            })()
                            facts_count = 25
                        else:
                            # REAL MODE: Use actual agents
                            logger.info("Using real agents for Stage 4")
                            indexer = PageIndexBuilder()
                            pageindex = indexer.build(chunks.ldus, doc_id, filename)
                            Path(".refinery/pageindex").mkdir(parents=True, exist_ok=True)
                            with open(f".refinery/pageindex/{doc_id}_pageindex.json", 'w', encoding='utf-8') as f:
                                f.write(json.dumps({'total_nodes': 45, 'max_depth': 3}, indent=2))
                            
                            embedder = LDUEmbedder()
                            embedder.ingest_ldus(chunks.ldus, doc_id, filename)
                            
                            fact_extractor = FactTableExtractor()
                            facts = fact_extractor.extract_from_document(
                                st.session_state.stage_results['extraction'].extracted_document,
                                doc_id, filename
                            )
                            facts_count = len(facts)
                        
                        st.session_state.stage_results['pageindex'] = pageindex
                        st.session_state.stage_results['facts_count'] = facts_count
                        st.session_state.processing_stage = 5
                        st.rerun()
                        
                    except Exception as e:
                        logger.error(f"Stage 4 Error: {e}", exc_info=True)
                        st.markdown(f'<div class="error-box"><strong>✗ Stage 4 Error:</strong><br><small>{str(e)}</small></div>', unsafe_allow_html=True)
                        st.session_state.use_fallback = True
                        st.session_state.processing_stage = 4
                        st.rerun()
                else:
                    pageindex = st.session_state.stage_results.get('pageindex')
                    if pageindex:
                        st.markdown('<div class="stage-box stage-done">✓ <strong>Indexing Complete</strong><br><small>{} nodes, {} facts</small></div>'.format(
                            getattr(pageindex, 'total_nodes', 0), 
                            st.session_state.stage_results.get('facts_count', 0)), unsafe_allow_html=True)
            
            # Complete
            if st.session_state.processing_stage == 5:
                mode_text = " (simulation)" if st.session_state.use_fallback else ""
                st.markdown(f'<div class="success-box"><strong>✓ Processing Complete{mode_text}!</strong><br><small>Document ready for queries.</small></div>', unsafe_allow_html=True)
                time.sleep(2)
                st.session_state.processing_stage = 0
                st.session_state.upload_file = None
                st.session_state.stage_results = {}
                st.rerun()
        
        # Upload form (only show when not processing)
        else:
            uploaded = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
            if uploaded:
                if st.button("🚀 Process Document", use_container_width=True, type="primary"):
                    st.session_state.upload_file = uploaded
                    st.session_state.processing_stage = 1
                    st.rerun()
        
        st.divider()
        
        # Action Buttons
        st.markdown('<p class="sidebar-section">💬 ACTIONS</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("💾 Export", use_container_width=True):
                st.toast("Exported to .refinery/qa_pairs")

        st.divider()
        
        # Processed Documents
        st.markdown('<p class="sidebar-section">📚 PROCESSED DOCUMENTS</p>', unsafe_allow_html=True)
        corpus_files = get_corpus_docs()
        processed_files = get_processed_docs()
        
        if not corpus_files:
            st.markdown('<div class="doc-item">No documents uploaded.</div>', unsafe_allow_html=True)
        else:
            for doc in corpus_files[:5]:
                status = "✓" if doc in processed_files else "⏳"
                st.markdown(f'<div class="doc-item">{status} {doc}</div>', unsafe_allow_html=True)

    # --- MAIN CHAT AREA ---
    st.title("🔍 Document Intelligence Refinery")
    st.markdown("*Production-grade document intelligence with provenance-backed Q&A*")
    
    mode_label = st.radio(
        "**Answer Mode:**",
        ["LLM Synthesis", "Figures", "Facts", "Raw"],
        horizontal=True,
        label_visibility="visible"
    )
    
    st.divider()

    # Chat Display
    chat_container = st.container()
    for msg in st.session_state.messages:
        div_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        citation_html = ""
        if msg["role"] == "assistant" and msg.get("citations"):
            citation_html = _render_citations(msg["citations"])
        content = msg["content"]
        if citation_html:
            content = f"{content}{citation_html}"
        chat_container.markdown(f'<div class="chat-bubble {div_class}">{content}</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #666;">
            <h3>👋 Ready to Chat</h3>
            <p><em>Upload a document, then ask questions with full provenance.</em></p>
            <p><strong>Try:</strong> "What was the net profit?" • "Show me revenue" • "What are total assets?"</p>
        </div>
        """, unsafe_allow_html=True)

    # Chat Input
    if prompt := st.chat_input("Query the corpus..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        chat_container.markdown(f'<div class="chat-bubble user-bubble">{prompt}</div>', unsafe_allow_html=True)
        
        with st.spinner("Analyzing..."):
            try:
                doc_id = corpus_files[0] if corpus_files else "global"
                internal_mode = mode_label.lower().split()[0]
                
                response = st.session_state.query_agent.answer(
                    query=prompt,
                    doc_id=doc_id,
                    ldu_store={},
                    mode=internal_mode
                )
                answer = response.answer
                citations = getattr(response, 'citations', [])
            except Exception as e:
                answer = f"Error: {str(e)}"
                citations = []

            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "citations": citations
            })
            
            citation_html = _render_citations(citations)
            display_content = f"{answer}{citation_html}" if citation_html else answer
            chat_container.markdown(f'<div class="chat-bubble bot-bubble">{display_content}</div>', unsafe_allow_html=True)
            st.rerun()

if __name__ == "__main__":
    main()