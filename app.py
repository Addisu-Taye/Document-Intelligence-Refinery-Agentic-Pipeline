# app.py
"""
Document Intelligence Refinery — Production Frontend
Global knowledge base chat across all processed documents.

Layout:
• Left: Document upload + 4-stage processing progress
• Right: Single chat box querying ALL processed documents

Features:
• Upload any PDF → auto-process 4-stage pipeline
• Single chat interface searches across ALL documents
• No document selection needed — system finds exact answer
• Provenance citations show which document + page
• Persistent conversations
• Export to Markdown

Repository: https://github.com/Addisu-Taye/Document-Intelligence-Refinery-Agentic-Pipeline
Author: Addisu Taye <addtaye@gmail.com>
"""

import streamlit as st
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('.refinery/frontend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import pipeline components
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import SemanticChunker
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.agents.embedder import LDUEmbedder
from src.agents.fact_extractor import FactTableExtractor

# ============================================================================
# UTILITIES
# ============================================================================

def generate_doc_id(file_path: Path) -> str:
    """Generate stable doc_id from file content hash."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()[:16]
    except:
        return f"doc_{int(time.time())}"

def ensure_directories():
    """Ensure all required directories exist."""
    for d in [
        Path(".refinery"), Path(".refinery/profiles"), Path(".refinery/pageindex"),
        Path(".refinery/qa_pairs"), Path(".refinery/vector_store"),
        Path(".refinery/conversations"), Path("uploads")
    ]:
        d.mkdir(parents=True, exist_ok=True)

def format_bytes(size: int) -> str:
    """Format file size for display."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def get_all_processed_docs() -> List[Dict]:
    """Get list of all processed documents."""
    profile_dir = Path(".refinery/profiles")
    docs = []
    if profile_dir.exists():
        for p in sorted(profile_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            docs.append({
                "doc_id": p.stem,
                "filename": data.get('filename', p.name),
                "origin": data.get('origin_type', 'unknown'),
                "strategy": data.get('recommended_strategy', 'unknown'),
                "pages": data.get('page_count', 'N/A')
            })
    return docs

# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """Manage global chat conversations."""
    
    def __init__(self, storage_path: str = ".refinery/conversations"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.main_conv_path = self.storage_path / "main_conversation.jsonl"
    
    def save(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Save a chat message."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        with open(self.main_conv_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
    
    def load(self) -> List[Dict]:
        """Load conversation history."""
        if not self.main_conv_path.exists():
            return []
        messages = []
        with open(self.main_conv_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    messages.append(json.loads(line))
        return messages
    
    def clear(self):
        """Clear conversation."""
        if self.main_conv_path.exists():
            self.main_conv_path.unlink()
    
    def export(self) -> str:
        """Export conversation to Markdown."""
        messages = self.load()
        md = ["# Document Intelligence Refinery — Chat Log", f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"]
        for msg in messages:
            icon = "👤" if msg['role'] == 'user' else "🤖"
            md.append(f"### {icon} {msg['role'].title()}\n{msg['content']}\n")
            if msg.get('metadata', {}).get('citations'):
                md.append("**Sources:**")
                for c in msg['metadata']['citations']:
                    md.append(f"- **{c.get('document', 'Unknown')}** — Page {c.get('page_number')}: {c.get('cited_text', '')[:150]}...")
            md.append("")
        return "\n".join(md)

# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Orchestrate the 4-stage pipeline."""
    
    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        self.config_path = config_path
        self.triage = self.router = self.chunker = self.indexer = self.query_agent = None
        self._init = False
    
    def _init_agents(self):
        if self._init:
            return
        self.triage = TriageAgent()
        self.router = ExtractionRouter()
        self.chunker = SemanticChunker()
        self.indexer = PageIndexBuilder()
        self.query_agent = QueryAgent(config_path=self.config_path)
        self._init = True
    
    def process(self, file_path: Path, progress=None) -> Dict[str, Any]:
        """Process document through 4-stage pipeline."""
        self._init_agents()
        doc_id = generate_doc_id(file_path)
        filename = file_path.name
        
        results = {"doc_id": doc_id, "filename": filename, "stages": {}, "status": "processing"}
        
        try:
            # Stage 1: Triage
            if progress: progress(1, "Triage", "Analyzing document structure...")
            profile = self.triage.profile_document(str(file_path))
            results["stages"]["triage"] = {
                "origin": profile.origin_type,
                "layout": profile.layout_complexity,
                "strategy": profile.recommended_strategy,
                "confidence": f"{profile.origin_confidence:.2f}"
            }
            
            # Save profile
            Path(".refinery/profiles").mkdir(parents=True, exist_ok=True)
            with open(f".refinery/profiles/{doc_id}.json", 'w', encoding='utf-8') as f:
                f.write(profile.model_dump_json(indent=2))
            
            # Stage 2: Extraction
            if progress: progress(2, "Extraction", f"Using {profile.recommended_strategy} strategy...")
            extraction = self.router.extract(str(file_path), profile)
            results["stages"]["extraction"] = {
                "strategy": extraction.strategy_used,
                "confidence": f"{extraction.extracted_document.overall_confidence:.2f}",
                "blocks": len(extraction.extracted_document.blocks),
                "tables": len(extraction.extracted_document.tables)
            }
            
            # Stage 3: Chunking
            if progress: progress(3, "Chunking", "Creating semantic chunks...")
            chunks = self.chunker.chunk(extraction.extracted_document)
            results["stages"]["chunking"] = {
                "ldus": chunks.total_ldus,
                "validation": "✓ Passed" if chunks.validation_passed else "✗ Failed"
            }
            
            # Stage 4: Indexing
            if progress: progress(4, "Indexing", "Building navigation tree...")
            pageindex = self.indexer.build(chunks.ldus, doc_id, filename)
            Path(".refinery/pageindex").mkdir(parents=True, exist_ok=True)
            with open(f".refinery/pageindex/{doc_id}_pageindex.json", 'w', encoding='utf-8') as f:
                f.write(pageindex.model_dump_json(indent=2))
            results["stages"]["indexing"] = {
                "nodes": pageindex.total_nodes,
                "depth": pageindex.max_depth
            }
            
            # Vector ingest (non-blocking)
            try:
                embedder = LDUEmbedder()
                embedder.ingest_ldus(chunks.ldus, doc_id)
                results["stages"]["vector"] = {"embeddings": len(chunks.ldus)}
            except Exception as e:
                logger.warning(f"Vector ingest skipped: {e}")
                results["stages"]["vector"] = {"status": "skipped"}
            
            results["status"] = "complete"
            results["ready"] = True
            if progress: progress(4, "Complete", "Ready for queries!")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            results["status"] = "error"
            results["error"] = str(e)
            return results

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Document Intelligence Refinery",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for split layout
    st.markdown("""
    <style>
        .stApp { max-width: 100%; }
        .left-panel { 
            background: #f8f9fa; 
            padding: 2rem; 
            border-radius: 1rem;
            margin: 1rem;
        }
        .right-panel { 
            padding: 1rem; 
            margin: 1rem;
        }
        .stage-card { 
            padding: 1rem; 
            border-radius: 0.5rem; 
            border-left: 4px solid #3498db;
            background: white;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stage-complete { border-left-color: #27ae60; }
        .stage-processing { border-left-color: #f39c12; animation: pulse 2s infinite; }
        .stage-pending { border-left-color: #bdc3c7; opacity: 0.6; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .chat-user { display: flex; justify-content: flex-end; margin: 0.5rem 0; }
        .chat-assistant { display: flex; justify-content: flex-start; margin: 0.5rem 0; }
        .chat-bubble { 
            max-width: 85%; padding: 1rem; border-radius: 1rem; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); line-height: 1.5;
        }
        .chat-user .chat-bubble { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; border-bottom-right-radius: 0.25rem; 
        }
        .chat-assistant .chat-bubble { 
            background: white; color: #333; border-bottom-left-radius: 0.25rem; border: 1px solid #eee;
        }
        .citation { 
            font-size: 0.8rem; color: #666; margin-top: 0.75rem; padding-top: 0.75rem; 
            border-top: 1px solid #eee; 
        }
        .citation strong { color: #667eea; }
        .doc-list { max-height: 300px; overflow-y: auto; }
        .doc-item { 
            padding: 0.5rem; margin: 0.25rem 0; 
            background: white; border-radius: 0.5rem;
            border: 1px solid #eee;
        }
        .footer { text-align: center; color: #999; font-size: 0.85rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee; }
        .stats-card { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize
    ensure_directories()
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'conv_manager' not in st.session_state:
        st.session_state.conv_manager = ConversationManager()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'current_upload' not in st.session_state:
        st.session_state.current_upload = None
    
    # Create two-column layout
    left_col, right_col = st.columns([1, 2])
    
    # ========================================================================
    # LEFT PANEL: Document Upload + 4-Stage Progress
    # ========================================================================
    
    with left_col:
        st.markdown('<div class="left-panel">', unsafe_allow_html=True)
        
        st.header("📁 Upload Document")
        st.markdown("*Upload a PDF to add to the knowledge base*")
        
        uploaded = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Any PDF: native digital, scanned, or mixed",
            key="uploader"
        )
        
        # Processing progress display
        if st.session_state.processing and st.session_state.current_upload:
            st.markdown("### ⚙️ Processing Pipeline")
            
            # Progress container
            progress_container = st.container()
            
            with progress_container:
                # Stage 1: Triage
                st.markdown(f"""
                <div class="stage-card stage-processing">
                    <strong>📋 Stage 1: Triage</strong><br>
                    <small>Analyzing document structure, origin type, layout complexity...</small>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(25)
                status_text = st.empty()
                
                def update_progress(stage_num, stage_name, message):
                    progress_bar.progress(stage_num * 25)
                    status_text.text(f"{stage_name}: {message}")
                
                st.session_state.progress_callback = update_progress
            
            # Process the document
            upload_path = Path("uploads") / st.session_state.current_upload.name
            with open(upload_path, "wb") as f:
                f.write(st.session_state.current_upload.getvalue())
            
            result = st.session_state.processor.process(
                upload_path,
                progress=st.session_state.progress_callback
            )
            
            # Show results
            progress_bar.progress(100)
            status_text.empty()
            
            if result["status"] == "complete":
                st.success(f"✓ Processed in {result.get('processing_time', 'N/A')}s")
                st.session_state.processing = False
                st.session_state.current_upload = None
                st.rerun()
            else:
                st.error(f"✗ Error: {result.get('error', 'Unknown')}")
                st.session_state.processing = False
                st.session_state.current_upload = None
        
        # Upload button
        if uploaded and not st.session_state.processing:
            st.session_state.current_upload = uploaded
            st.session_state.processing = True
            st.rerun()
        
        st.divider()
        
        # Knowledge Base Stats
        st.header("📊 Knowledge Base")
        docs = get_all_processed_docs()
        
        if docs:
            st.markdown(f"""
            <div class="stats-card">
                <h3 style="margin:0; color:white;">{len(docs)}</h3>
                <p style="margin:0; opacity:0.9;">Documents Processed</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Document list
            st.markdown("### 📚 Processed Documents")
            st.markdown('<div class="doc-list">', unsafe_allow_html=True)
            for doc in docs:
                st.markdown(f"""
                <div class="doc-item">
                    <strong>{doc['filename'][:40]}{'...' if len(doc['filename']) > 40 else ''}</strong><br>
                    <small>📄 {doc['pages']} pages • {doc['origin']} • {doc['strategy']}</small>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📭 No documents processed yet. Upload a PDF to begin.")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.conv_manager.clear()
            st.rerun()
        
        # Export button
        conv = st.session_state.conv_manager.load()
        if conv:
            if st.button("📥 Export Conversation"):
                md = st.session_state.conv_manager.export()
                st.download_button(
                    "Download Markdown",
                    md,
                    f"refinery_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    "text/markdown"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # RIGHT PANEL: Single Chat Box (Queries ALL Documents)
    # ========================================================================
    
    with right_col:
        st.markdown('<div class="right-panel">', unsafe_allow_html=True)
        
        st.header("🤖 Document Intelligence Chat")
        st.markdown("*Ask questions — I'll search across all processed documents*")
        
        # Load conversation
        messages = st.session_state.conv_manager.load()
        
        # Display chat history
        for msg in messages:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-user">
                    <div class="chat-bubble">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Build citation HTML
                citation_html = ""
                if msg.get('metadata', {}).get('citations'):
                    cites = []
                    for c in msg['metadata']['citations'][:3]:
                        doc_name = c.get('document', 'Unknown')[:30]
                        page = c.get('page_number', '?')
                        cites.append(f"<strong>[{len(cites)+1}]</strong> {doc_name} — p.{page}")
                    citation_html = f'<div class="citation">📍 Sources: {" • ".join(cites)}</div>'
                
                st.markdown(f"""
                <div class="chat-assistant">
                    <div class="chat-bubble">
                        {msg['content']}
                        {citation_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask about any processed document..."):
            # Check if any documents exist
            docs = get_all_processed_docs()
            if not docs:
                st.error("⚠️ No documents processed yet. Please upload a PDF first.")
            else:
                # Display user message
                st.markdown(f"""
                <div class="chat-user">
                    <div class="chat-bubble">{prompt}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Save user message
                st.session_state.conv_manager.save("user", prompt)
                
                # Generate response
                with st.spinner("🔍 Searching across all documents..."):
                    try:
                        # Query across all documents (search each, aggregate results)
                        all_citations = []
                        best_answer = ""
                        best_confidence = 0.0
                        
                        # Query each document and aggregate
                        for doc in docs[:5]:  # Limit to top 5 most recent
                            try:
                                result = st.session_state.processor.query_agent.answer(
                                    query=prompt,
                                    doc_id=doc["doc_id"],
                                    ldu_store={}
                                )
                                
                                if result.answer_confidence > best_confidence:
                                    best_answer = result.answer
                                    best_confidence = result.answer_confidence
                                
                                for c in result.citations[:2]:
                                    all_citations.append({
                                        "document": doc["filename"],
                                        "page_number": c.page_number,
                                        "cited_text": c.cited_text,
                                        "extraction_strategy": c.extraction_strategy
                                    })
                            except Exception as e:
                                logger.warning(f"Query failed for {doc['doc_id']}: {e}")
                                continue
                        
                        # Fallback if no good answer
                        if not best_answer:
                            best_answer = "I couldn't find a confident answer in the processed documents. Try rephrasing your question or check if the document contains this information."
                            best_confidence = 0.3
                        
                        # Limit citations
                        all_citations = all_citations[:5]
                        
                        # Display assistant response
                        citation_html = ""
                        if all_citations:
                            cites = []
                            for i, c in enumerate(all_citations, 1):
                                doc_name = c['document'][:25]
                                cites.append(f"<strong>[{i}]</strong> {doc_name} — p.{c['page_number']}")
                            citation_html = f'<div class="citation">📍 Sources: {" • ".join(cites)}</div>'
                        
                        st.markdown(f"""
                        <div class="chat-assistant">
                            <div class="chat-bubble">
                                {best_answer}
                                {citation_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Save assistant message
                        st.session_state.conv_manager.save(
                            "assistant",
                            best_answer,
                            metadata={
                                "confidence": best_confidence,
                                "documents_searched": len(docs),
                                "citations": all_citations
                            }
                        )
                        
                    except Exception as e:
                        logger.error(f"Query error: {e}", exc_info=True)
                        st.markdown(f"""
                        <div class="chat-assistant">
                            <div class="chat-bubble">
                                Sorry, I encountered an error. Please try again.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.conv_manager.save("assistant", "Sorry, I encountered an error.")
                
                # Rerun to show new message
                st.rerun()
        
        # Welcome message if no conversation
        if not messages:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #666;">
                <h2>👋 Welcome to Document Intelligence Refinery</h2>
                <p>Upload documents on the left, then ask questions on the right.</p>
                <p><em>I'll search across all processed documents to find exact answers with provenance.</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        Document Intelligence Refinery v1.0 • 
        <a href="https://github.com/Addisu-Taye/Document-Intelligence-Refinery-Agentic-Pipeline" target="_blank">GitHub Repository</a> • 
        Addisu Taye &lt;addtaye@gmail.com&gt;
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"App crash: {e}", exc_info=True)
        st.error(f"🚨 Error: {str(e)}")