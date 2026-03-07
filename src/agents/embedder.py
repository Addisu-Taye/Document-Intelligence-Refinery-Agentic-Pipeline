# src/agents/embedder.py — UNIVERSAL VERSION (No Model Imports)
"""
LDU Embedder — Universal version without model dependencies.
Adds filename to metadata for proper citations.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('.refinery/embedder.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LDUEmbedder:
    """Embed LDUs into ChromaDB vector store with complete metadata."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_path: str = ".refinery/vector_store"):
        self.model_name = model_name
        self.persist_path = persist_path
        
        print(f"  Vector store: {Path(persist_path).resolve()}")
        print(f"  Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("  ✓ Model loaded")
        
        print(f"  Initializing ChromaDB 0.5.x PersistentClient...")
        self.client = chromadb.PersistentClient(
            path=str(Path(persist_path).resolve()),
            settings=Settings(allow_reset=True)
        )
        print("  ✓ Client initialized")
        
        self.collection = self.client.get_or_create_collection(
            name="ldu_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        print("  ✓ Collection: ldu_embeddings")
        print(f"  ✓ Current embeddings: {self.collection.count()}")
    
    def _ldu_to_metadata(self, ldu: Any, doc_id: str, filename: str) -> Dict:
        """Convert LDU to ChromaDB metadata WITH filename field."""
        # Use getattr for duck-typing compatibility
        return {
            "ldu_id": getattr(ldu, 'ldu_id', f"{doc_id}_{hash(str(ldu)) % 100000}"),
            "doc_id": doc_id,
            "filename": filename,  # ← ADD THIS FIELD
            "page_refs": json.dumps(getattr(ldu, 'page_refs', [])) if hasattr(ldu, 'page_refs') else '[]',
            "chunk_type": getattr(ldu, 'chunk_type', 'text'),
            "section_path": " > ".join(getattr(ldu, 'section_path', [])) if hasattr(ldu, 'section_path') and ldu.section_path else "",
            "token_count": getattr(ldu, 'token_count', 0),
            "extraction_strategy": getattr(ldu, 'extraction_strategy', 'unknown')
        }
    
    def ingest_ldus(self, ldus: List[Any], doc_id: str, filename: str):
        """Ingest LDUs into vector store with complete metadata."""
        if not ldus:
            logger.warning("No LDUs to ingest")
            return
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for ldu in ldus:
            # Get content using getattr for compatibility
            content = getattr(ldu, 'content', None) or getattr(ldu, 'text', None) or str(ldu)
            
            # Skip empty LDUs
            if not content or not str(content).strip():
                continue
            
            ldu_id = getattr(ldu, 'ldu_id', f"{doc_id}_{hash(str(content)) % 100000}")
            full_id = f"{doc_id}_{ldu_id}"
            
            # Avoid duplicates
            if full_id in ids:
                continue
            
            ids.append(full_id)
            embeddings.append(self.model.encode(str(content)).tolist())
            metadatas.append(self._ldu_to_metadata(ldu, doc_id, filename))
            documents.append(str(content))
        
        if ids:
            # Delete existing entries for this doc_id to avoid duplicates
            existing = self.collection.get(where={"doc_id": doc_id}, include=[])
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
            
            # Batch insert (ChromaDB limit: ~1000 per batch)
            batch_size = 500
            for i in range(0, len(ids), batch_size):
                self.collection.add(
                    ids=ids[i:i+batch_size],
                    embeddings=embeddings[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    documents=documents[i:i+batch_size]
                )
            
            logger.info(f"✓ Ingested {len(ids)} embeddings for {filename}")
    
    def search(self, query: str, doc_id: Optional[str] = None, n_results: int = 15) -> Dict[str, Any]:
        """Search vector store — GLOBAL by default, optional doc_id filter."""
        query_embedding = self.model.encode([query]).tolist()
        
        # Don't filter by doc_id by default — search ALL documents
        where_filter = None
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,  # ← Increased from 5 to 15 for better recall
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        return {
            "total_embeddings": self.collection.count(),
            "collection_name": self.collection.name
        }


# ============================================================================
# CLI for re-ingestion
# ============================================================================

if __name__ == "__main__":
    """Re-ingest vectors for all processed documents."""
    import sys
    sys.path.insert(0, '.')
    
    # Import only what we need (no chunker dependency)
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from pathlib import Path
    
    print("🔄 Re-ingesting vectors for all documents...")
    
    embedder = LDUEmbedder()
    triage = TriageAgent()
    router = ExtractionRouter()
    
    # Import chunker inside try block in case it fails
    try:
        from src.agents.chunker import SemanticChunker
        chunker = SemanticChunker()
        has_chunker = True
    except Exception as e:
        print(f"⚠️  Chunker import failed: {e}")
        print("   Will use existing LDUs from extraction")
        has_chunker = False
    
    profile_dir = Path(".refinery/profiles")
    if not profile_dir.exists():
        print("❌ No processed documents found")
        sys.exit(1)
    
    for profile_path in profile_dir.glob("*.json"):
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        
        doc_id = profile_path.stem
        filename = profile_data.get('filename', '')
        
        pdf_path = Path("corpus") / filename
        if not pdf_path.exists():
            print(f"⚠️  Source not found: {filename}")
            continue
        
        print(f"\n📄 Processing {filename}...")
        
        try:
            # Re-process: extract → chunk → embed
            profile = triage.profile_document(str(pdf_path))
            extraction = router.extract(str(pdf_path), profile)
            
            if has_chunker:
                chunks = chunker.chunk(extraction.extracted_document)
                embedder.ingest_ldus(chunks.ldus, doc_id, filename)
                print(f"   ✓ Ingested {chunks.total_ldus} LDUs")
            else:
                # Fallback: use blocks as LDUs
                blocks = getattr(extraction.extracted_document, 'blocks', [])
                embedder.ingest_ldus(blocks, doc_id, filename)
                print(f"   ✓ Ingested {len(blocks)} blocks as LDUs")
        
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    stats = embedder.get_stats()
    print(f"\n✅ Vector store: {stats['total_embeddings']} total embeddings")