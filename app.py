# app.py
"""
Document Intelligence Refinery — Production Frontend
Single chat interface for all processed documents.

Features:
• Upload any PDF → auto-process pipeline
• Single chat box (standard chatbot UI) for all documents
• Document selector dropdown to switch context
• Persistent conversations per document
• Provenance citations inline with answers
• Export conversation to Markdown
• Professional, clean UI

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

# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """Manage chat conversations with persistence per document."""
    
    def __init__(self, storage_path: str = ".refinery/conversations"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, doc_id: str) -> Path:
        return self.storage_path / f"{doc_id}.jsonl"
    
    def save(self, doc_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Save a chat message."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        with open(self._get_path(doc_id), 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
    
    def load(self, doc_id: str) -> List[Dict]:
        """Load conversation history for a document."""
        path = self._get_path(doc_id)
        if not path.exists():
            return []
        messages = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    messages.append(json.loads(line))
        return messages
    
    def clear(self, doc_id: str):
        """Clear conversation for a document."""
        path = self._get_path(doc_id)
        if path.exists():
            path.unlink()
    
    def export(self, doc_id: str, filename: str) -> str:
        """Export conversation to Markdown."""
        messages = self.load(doc_id)
        md = [f"# Chat: {filename}", f"*Document ID: {doc_id}*\n"]
        for msg in messages:
            icon = "👤" if msg['role'] == 'user' else "🤖"
            md.append(f"### {icon} {msg['role'].title()}\n{msg['content']}\n")
            if msg.get('metadata', {}).get('citations'):
                md.append("**Sources:**")
                for c in msg['metadata']['citations']:
                    md.append(f"- Page {c.get('page_number')}: {c.get('cited_text', '')[:150]}...")
            md.append("")
        return "\n".join(md)

# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Orchestrate the 5-stage pipeline."""
    
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
        """Process document through full pipeline."""
        self._init_agents()
        doc_id = generate_doc_id(file_path)
        filename = file_path.name
        
        results = {"doc_id": doc_id, "filename": filename, "stages": {}, "artifacts": {}, "status": "processing"}
        
        try:
            # Stage 1: Triage
            if progress: progress("Triage", 20, "Analyzing structure...")
            profile = self.triage.profile_document(str(file_path))
            results["stages"]["triage"] = {"origin": profile.origin_type, "strategy": profile.recommended_strategy}
            
            # Save profile
            Path(".refinery/profiles").mkdir(parents=True, exist_ok=True)
            with open(f".refinery/profiles/{doc_id}.json", 'w') as f:
                f.write(profile.model_dump_json(indent=2))
            
            # Stage 2: Extraction
            if progress: progress("Extraction", 40, f"Using {profile.recommended_strategy}...")
            extraction = self.router.extract(str(file_path), profile)
            results["stages"]["extraction"] = {"strategy": extraction.strategy_used, "confidence": extraction.extracted_document.overall_confidence}
            
            # Stage 3: Chunking
            if progress: progress("Chunking", 60, "Creating semantic chunks...")
            chunks = self.chunker.chunk(extraction.extracted_document)
            results["stages"]["chunking"] = {"ldus": chunks.total_ldus}
            
            # Stage 4: PageIndex
            if progress: progress("Indexing", 80, "Building navigation tree...")
            pageindex = self.indexer.build(chunks.ldus, doc_id, filename)
            Path(".refinery/pageindex").mkdir(parents=True, exist_ok=True)
            with open(f".refinery/pageindex/{doc_id}_pageindex.json", 'w') as f:
                f.write(pageindex.model_dump_json(indent=2))
            results["stages"]["indexing"] = {"nodes": pageindex.total_nodes}
            
            # Stage 5: Vector (non-blocking)
            try:
                if progress: progress("Vector", 95, "Generating embeddings...")
                embedder = LDUEmbedder()
                embedder.ingest_ldus(chunks.ldus, doc_id)
            except Exception as e:
                logger.warning(f"Vector ingest skipped: {e}")
            
            results["status"] = "complete"
            results["ready"] = True
            if progress: progress("Done", 100, "Ready to chat!")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            return results

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="Document Intelligence Refinery", page_icon="🔍", layout="wide")
    
    # Custom CSS for standard chatbot UI
    st.markdown("""
    <style>
        .stApp { max-width: 1000px; margin: 0 auto; }
        .chat-user { display: flex; justify-content: flex-end; margin: 0.5rem 0; }
        .chat-assistant { display: flex; justify-content: flex-start; margin: 0.5rem 0; }
        .chat-bubble { 
            max-width: 80%; padding: 0.75rem 1rem; border-radius: 1rem; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
        }
        .chat-user .chat-bubble { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; border-bottom-right-radius: 0.25rem; 
        }
        .chat-assistant .chat-bubble { 
            background: #f8f9fa; color: #333; border-bottom-left-radius: 0.25rem; 
        }
        .citation { 
            font-size: 0.85rem; color: #666; margin-top: 0.5rem; padding-top: 0.5rem; 
            border-top: 1px solid #eee; 
        }
        .citation a { color: #667eea; text-decoration: none; }
        .doc-select { margin: 1rem 0; }
        .footer { text-align: center; color: #999; font-size: 0.85rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🔍 Document Intelligence Refinery")
    
    # Initialize
    ensure_directories()
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'conv_manager' not in st.session_state:
        st.session_state.conv_manager = ConversationManager()
    if 'selected_doc' not in st.session_state:
        st.session_state.selected_doc = None
    
    # Sidebar: Upload & Document List
    with st.sidebar:
        st.header("📁 Upload Document")
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"], help="Any PDF: digital, scanned, or mixed")
        
        if uploaded:
            upload_path = Path("uploads") / uploaded.name
            with open(upload_path, "wb") as f:
                f.write(uploaded.getvalue())
            
            if st.button("🚀 Process", type="primary"):
                with st.spinner("Processing document..."):
                    def progress(stage, pct, msg):
                        st.toast(f"{stage}: {msg}")
                    
                    result = st.session_state.processor.process(upload_path, progress)
                    
                    if result["status"] == "complete":
                        st.success(f"✓ Processed in {result.get('processing_time', 'N/A')}s")
                        # Auto-select the new document
                        st.session_state.selected_doc = {
                            "doc_id": result["doc_id"],
                            "filename": result["filename"],
                            "profile": result["stages"].get("triage", {})
                        }
                        st.rerun()
                    else:
                        st.error(f"✗ Error: {result.get('error', 'Unknown')}")
        
        st.divider()
        st.header("📚 Processed Documents")
        
        # List processed documents
        profile_dir = Path(".refinery/profiles")
        docs = []
        if profile_dir.exists():
            for p in sorted(profile_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                with open(p, 'r') as f:
                    data = json.load(f)
                docs.append({"doc_id": p.stem, "filename": data.get('filename', p.name), "origin": data.get('origin_type', 'unknown')})
        
        if docs:
            # Document selector dropdown
            doc_options = [f"{d['filename']} ({d['origin']})" for d in docs]
            selected_label = st.selectbox("Select document to chat with", doc_options, index=0, key="doc_selector")
            
            if st.button("💬 Start Chatting", type="primary"):
                selected = docs[doc_options.index(selected_label)]
                st.session_state.selected_doc = {
                    "doc_id": selected["doc_id"],
                    "filename": selected["filename"],
                    "profile": {}
                }
                st.rerun()
        else:
            st.info("Upload and process a document to begin.")
        
        st.divider()
        if st.button("🗑️ Clear Current Chat"):
            if st.session_state.selected_doc:
                st.session_state.conv_manager.clear(st.session_state.selected_doc["doc_id"])
                st.rerun()

    # Main: Single Chat Interface
    if st.session_state.selected_doc:
        doc = st.session_state.selected_doc
        
        # Document header
        st.markdown(f"### 💬 Chatting with: **{doc['filename']}**")
        st.caption(f"Document ID: `{doc['doc_id']}` | Origin: {doc['profile'].get('origin', 'unknown')}")
        
        # Load conversation
        conv = st.session_state.conv_manager.load(doc["doc_id"])
        
        # Display chat history (standard chatbot style)
        for msg in conv:
            if msg['role'] == 'user':
                st.markdown(f'<div class="chat-user"><div class="chat-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                citation_html = ""
                if msg.get('metadata', {}).get('citations'):
                    cites = []
                    for i, c in enumerate(msg['metadata']['citations'][:2], 1):  # Show first 2
                        cites.append(f"<a href=\"#\" title=\"Page {c.get('page_number')}\">[{i}] p.{c.get('page_number')}</a>")
                    citation_html = f'<div class="citation">Sources: {" • ".join(cites)}</div>'
                st.markdown(f'<div class="chat-assistant"><div class="chat-bubble">{msg["content"]}{citation_html}</div></div>', unsafe_allow_html=True)
        
        # Chat input (standard chatbot input at bottom)
        if prompt := st.chat_input("Ask about this document..."):
            # Display user message immediately
            st.markdown(f'<div class="chat-user"><div class="chat-bubble">{prompt}</div></div>', unsafe_allow_html=True)
            
            # Save user message
            st.session_state.conv_manager.save(doc["doc_id"], "user", prompt)
            
            # Generate assistant response
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.processor.query_agent.answer(
                        query=prompt,
                        doc_id=doc["doc_id"],
                        ldu_store={}
                    )
                    
                    # Build citation display
                    citations = []
                    for c in result.citations[:3]:  # Limit to 3 for clean UI
                        citations.append({
                            "page_number": c.page_number,
                            "cited_text": c.cited_text,
                            "extraction_strategy": c.extraction_strategy
                        })
                    
                    # Display assistant message
                    citation_html = ""
                    if citations:
                        cites = []
                        for i, c in enumerate(citations, 1):
                            cites.append(f"<a href=\"#\" title=\"Page {c['page_number']}: {c['cited_text'][:100]}...\">[{i}] p.{c['page_number']}</a>")
                        citation_html = f'<div class="citation">Sources: {" • ".join(cites)}</div>'
                    
                    st.markdown(f'<div class="chat-assistant"><div class="chat-bubble">{result.answer}{citation_html}</div></div>', unsafe_allow_html=True)
                    
                    # Save assistant message
                    st.session_state.conv_manager.save(
                        doc["doc_id"], "assistant", result.answer,
                        metadata={"confidence": result.answer_confidence, "citations": citations}
                    )
                    
                except Exception as e:
                    logger.error(f"Query error: {e}")
                    st.markdown(f'<div class="chat-assistant"><div class="chat-bubble">Sorry, I encountered an error. Please try again.</div></div>', unsafe_allow_html=True)
                    st.session_state.conv_manager.save(doc["doc_id"], "assistant", "Sorry, I encountered an error.")
            
            # Auto-scroll (rerun to show new message)
            st.rerun()
        
        # Export button
        if conv:
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("📥 Export"):
                    md = st.session_state.conv_manager.export(doc["doc_id"], doc["filename"])
                    st.download_button("Download Markdown", md, f"{doc['filename']}_chat.md", "text/markdown")
    
    else:
        # Welcome screen when no document selected
        st.markdown("""
        ### Welcome to Document Intelligence Refinery
        
        **Get started:**
        1. Upload a PDF in the sidebar
        2. Wait for auto-processing to complete
        3. Select your document from the dropdown
        4. Start chatting with your document!
        
        **Features:**
        • ✅ Auto-runs 5-stage pipeline (Triage → Extraction → Chunking → Indexing → Vector)
        • ✅ Single chat interface for all documents
        • ✅ Provenance citations with page numbers
        • ✅ Conversations persist per document
        • ✅ Export chats to Markdown for sharing
        
        *Upload a document to begin.*
        """)
    
    # Footer
    st.markdown('<div class="footer">Document Intelligence Refinery v1.0 • <a href="https://github.com/Addisu-Taye/Document-Intelligence-Refinery-Agentic-Pipeline" target="_blank">GitHub</a> • Addisu Taye</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"App crash: {e}", exc_info=True)
        st.error(f"🚨 Error: {str(e)}")