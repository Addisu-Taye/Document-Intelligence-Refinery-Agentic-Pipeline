# src/agents/embedder.py
"""
Vector Store Ingestion - ChromaDB 0.5.x Compatible
Fixed for proper disk persistence with ChromaDB 0.5.23
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timezone
import json

from src.models.ldu import LDU


class LDUEmbedder:
    """Generate embeddings for LDUs and store in ChromaDB 0.5.x with proper persistence."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_dir: str = ".refinery/vector_store"):
        """Initialize embedder with ChromaDB 0.5.x persistence."""
        self.model_name = model_name
        self.persist_dir = Path(persist_dir).resolve()  # Absolute path is critical
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Vector store: {self.persist_dir}", flush=True)
        
        # Load embedding model
        print(f"  Loading model: {model_name}...", flush=True)
        self.model = SentenceTransformer(model_name)
        print(f"  ✓ Model loaded", flush=True)
        
        # ChromaDB 0.5.x: Use PersistentClient with explicit path
        print(f"  Initializing ChromaDB 0.5.x PersistentClient...", flush=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        print(f"  ✓ Client initialized", flush=True)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="ldu_embeddings",
            metadata={
                "description": "LDU embeddings for RAG",
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        )
        print(f"  ✓ Collection: {self.collection.name}", flush=True)
        
        # Verify collection is accessible
        count = self.collection.count()
        print(f"  ✓ Current embeddings: {count}", flush=True)
    
    def _ldu_to_metadata(self, ldu: LDU, doc_id: str) -> Dict:
        """Convert LDU to ChromaDB-compatible metadata."""
        return {
            "ldu_id": ldu.ldu_id,
            "doc_id": doc_id,
            "page_refs": json.dumps(ldu.page_refs),
            "chunk_type": ldu.chunk_type,
            "section_path": " > ".join(ldu.section_path) if ldu.section_path else "",
            "token_count": ldu.token_count,
            "extraction_strategy": ldu.extraction_strategy
        }
    
    def ingest_ldus(self, ldus: List[LDU], doc_id: str, batch_size: int = 50) -> int:
        """Ingest LDUs into ChromaDB."""
        if not ldus:
            return 0
        
        ids = []
        documents = []
        metadatas = []
        
        for ldu in ldus:
            if not ldu.content or len(ldu.content.strip()) < 10:
                continue
            
            ldu_id = f"{doc_id}_{ldu.ldu_id}"
            
            # Skip duplicates
            try:
                existing = self.collection.get(ids=[ldu_id])
                if existing['ids']:
                    continue
            except:
                pass
            
            ids.append(ldu_id)
            documents.append(ldu.content)
            metadatas.append(self._ldu_to_metadata(ldu, doc_id))
        
        if not ids:
            return 0
        
        print(f"    Ingesting {len(ids)} LDUs...", flush=True)
        
        # Process in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            
            # Generate embeddings
            embeddings = self.model.encode(batch_docs, show_progress_bar=False)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                ids=batch_ids,
                metadatas=batch_meta,
                documents=batch_docs
            )
        
        # ChromaDB 0.5.x: Explicit persist
        try:
            self.client.persist()
            print(f"    ✓ Data persisted to disk", flush=True)
        except AttributeError:
            # persist() may not exist in all 0.5.x versions
            pass
        
        return len(ids)
    
    def search(self, query: str, doc_id: Optional[str] = None, n_results: int = 5) -> Dict:
        """Semantic search for relevant LDUs."""
        query_embedding = self.model.encode([query])
        where_filter = {"doc_id": doc_id} if doc_id else None
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "total_embeddings": self.collection.count(),
            "persist_directory": str(self.persist_dir),
            "collection_name": self.collection.name,
            "model": self.model_name,
            "chromadb_version": chromadb.__version__
        }


def ingest_all_corpus(corpus_dir: str = "corpus", output_dir: str = ".refinery"):
    """Ingest all corpus documents into vector store."""
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from src.agents.chunker import SemanticChunker
    
    corpus_path = Path(corpus_dir)
    output_path = Path(output_dir)
    
    print("=" * 60, flush=True)
    print("VECTOR STORE INGESTION (ChromaDB 0.5.23)", flush=True)
    print("=" * 60, flush=True)
    
    # Initialize embedder FIRST (critical for persistence)
    print("\nInitializing embedder...", flush=True)
    embedder = LDUEmbedder()
    
    triage = TriageAgent()
    chunker = SemanticChunker()
    
    pdf_files = list(corpus_path.glob("*.pdf"))
    print(f"\n📚 Ingesting {len(pdf_files)} documents...", flush=True)
    print("=" * 60, flush=True)
    
    total_ldus = 0
    
    for pdf_path in pdf_files:
        print(f"\n📄 Processing: {pdf_path.name}", flush=True)
        
        try:
            doc_id = pdf_path.stem.replace(" ", "_").lower()[:32]
            
            # Profile
            profile = triage.profile_document(str(pdf_path))
            
            # Extract
            router = ExtractionRouter()
            extraction_result = router.extract(str(pdf_path), profile)
            
            # Chunk
            chunk_result = chunker.chunk(extraction_result.extracted_document)
            print(f"     LDUs: {chunk_result.total_ldus}", flush=True)
            
            # Ingest
            count = embedder.ingest_ldus(chunk_result.ldus, doc_id)
            total_ldus += count
            
            print(f"  ✓ Ingested {count} LDUs", flush=True)
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60, flush=True)
    print(f"✅ Total LDUs ingested: {total_ldus}", flush=True)
    print(f"📍 Vector store: {embedder.persist_dir}", flush=True)
    
    # Save stats
    stats = embedder.get_stats()
    stats_path = output_path / "vector_store_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"📊 Stats saved to: {stats_path}", flush=True)
    print("\n🎉 Ingestion Complete!", flush=True)
    
    return total_ldus


if __name__ == "__main__":
    ingest_all_corpus()
