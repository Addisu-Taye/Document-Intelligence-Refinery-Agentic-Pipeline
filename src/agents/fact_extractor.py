# src/agents/fact_extractor.py
"""
FactTable Extractor with SQLite Backend

Extracts key-value facts from financial/numerical documents for precise SQL queries.
Required for Final Submission (Data Layer).
"""

import sqlite3
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from src.models.ldu import LDU


class Fact(BaseModel):
    """A single extracted fact with provenance."""
    fact_id: str
    entity: str  # e.g., "revenue", "net_profit", "total_expense"
    value: str   # e.g., "ETB 14.2 billion"
    numeric_value: Optional[float] = None
    unit: Optional[str] = None  # e.g., "billion", "million", "%"
    period: Optional[str] = None  # e.g., "FY 2023-24", "Q3 2024"
    doc_id: str
    page_refs: List[int]
    content_hash: str
    extraction_confidence: float


class FactTableExtractor:
    """Extract facts from LDUs and store in SQLite."""
    
    # Patterns for financial fact extraction
    PATTERNS = {
        'revenue': r'(?:revenue|income|sales)[\s:]+(?:ETB|Birr|\$)?\s*([\d,\.]+)\s*(billion|million|thousand)?',
        'net_profit': r'(?:net\s*profit|net\s*income|profit\s*after\s*tax)[\s:]+(?:ETB|Birr|\$)?\s*([\d,\.]+)\s*(billion|million|thousand)?',
        'total_expense': r'(?:total\s*expense|total\s*expenditure|operating\s*expense)[\s:]+(?:ETB|Birr|\$)?\s*([\d,\.]+)\s*(billion|million|thousand)?',
        'assets': r'(?:total\s*assets|assets)[\s:]+(?:ETB|Birr|\$)?\s*([\d,\.]+)\s*(billion|million|thousand)?',
        'equity': r'(?:total\s*equity|shareholders?\s*equity)[\s:]+(?:ETB|Birr|\$)?\s*([\d,\.]+)\s*(billion|million|thousand)?',
    }
    
    def __init__(self, db_path: str = ".refinery/facts.db"):
        """Initialize SQLite database for facts."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                entity TEXT NOT NULL,
                value TEXT NOT NULL,
                numeric_value REAL,
                unit TEXT,
                period TEXT,
                doc_id TEXT NOT NULL,
                page_refs TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                extraction_confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity ON facts(entity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_id ON facts(doc_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_period ON facts(period)')
        
        conn.commit()
        conn.close()
    
    def extract_facts(self, ldus: List[LDU], doc_id: str) -> List[Fact]:
        """Extract facts from LDUs."""
        facts = []
        fact_counter = 0
        
        for ldu in ldus:
            if ldu.chunk_type != "text_block" or not ldu.content:
                continue
            
            content = ldu.content.lower()
            
            for entity, pattern in self.PATTERNS.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    fact_counter += 1
                    fact_id = f"{doc_id}_{entity}_{fact_counter}"
                    
                    # Parse value
                    numeric_str = match.group(1).replace(',', '')
                    try:
                        numeric_value = float(numeric_str)
                    except:
                        numeric_value = None
                    
                    unit = match.group(2) if match.lastindex >= 2 else None
                    
                    fact = Fact(
                        fact_id=fact_id,
                        entity=entity,
                        value=match.group(0),
                        numeric_value=numeric_value,
                        unit=unit,
                        period=self._extract_period(content),
                        doc_id=doc_id,
                        page_refs=ldu.page_refs,
                        content_hash=ldu.content_hash,
                        extraction_confidence=ldu.extraction_confidence
                    )
                    facts.append(fact)
        
        return facts
    
    def _extract_period(self, text: str) -> Optional[str]:
        """Extract fiscal period from text."""
        patterns = [
            r'(?:FY|fiscal\s*year)\s*[\d]{4}[-/][\d]{2,4}',
            r'[\d]{4}[-/][\d]{2,4}',
            r'(?:Q[1-4]\s*[\d]{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def store_facts(self, facts: List[Fact]):
        """Store facts in SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for fact in facts:
            cursor.execute('''
                INSERT OR REPLACE INTO facts 
                (fact_id, entity, value, numeric_value, unit, period, doc_id, page_refs, content_hash, extraction_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fact.fact_id,
                fact.entity,
                fact.value,
                fact.numeric_value,
                fact.unit,
                fact.period,
                fact.doc_id,
                json.dumps(fact.page_refs),
                fact.content_hash,
                fact.extraction_confidence
            ))
        
        conn.commit()
        conn.close()
    
    def query_facts(self, entity: Optional[str] = None, doc_id: Optional[str] = None, 
                   period: Optional[str] = None) -> List[Dict]:
        """Query facts from SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM facts WHERE 1=1"
        params = []
        
        if entity:
            query += " AND entity = ?"
            params.append(entity)
        
        if doc_id:
            query += " AND doc_id = ?"
            params.append(doc_id)
        
        if period:
            query += " AND period LIKE ?"
            params.append(f"%{period}%")
        
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_stats(self) -> Dict:
        """Get fact table statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM facts")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT doc_id) as docs FROM facts")
        docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT entity) as entities FROM facts")
        entities = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_facts": total,
            "documents": docs,
            "entity_types": entities,
            "db_path": str(self.db_path)
        }


def extract_all_corpus(corpus_dir: str = "corpus", output_dir: str = ".refinery"):
    """Extract facts from all corpus documents."""
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from src.agents.chunker import SemanticChunker
    
    corpus_path = Path(corpus_dir)
    output_path = Path(output_dir)
    
    print("=" * 60)
    print("FACTTABLE EXTRACTION (Final Submission Requirement)")
    print("=" * 60)
    
    # Initialize components
    extractor = FactTableExtractor(db_path=str(output_path / "facts.db"))
    triage = TriageAgent()
    router = ExtractionRouter()
    chunker = SemanticChunker()
    
    pdf_files = list(corpus_path.glob("*.pdf"))
    print(f"\n📚 Extracting facts from {len(pdf_files)} documents...")
    print("=" * 60)
    
    total_facts = 0
    
    for pdf_path in pdf_files:
        print(f"\n📄 {pdf_path.name}...")
        
        try:
            doc_id = pdf_path.stem.replace(" ", "_").lower()[:32]
            
            # Full pipeline
            profile = triage.profile_document(str(pdf_path))
            extraction = router.extract(str(pdf_path), profile)
            chunks = chunker.chunk(extraction.extracted_document)
            
            # Extract facts
            facts = extractor.extract_facts(chunks.ldus, doc_id)
            extractor.store_facts(facts)
            
            total_facts += len(facts)
            print(f"  ✓ Extracted {len(facts)} facts")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"✅ Total facts extracted: {total_facts}")
    
    # Save stats
    stats = extractor.get_stats()
    stats_path = output_path / "facttable_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📊 Stats saved to: {stats_path}")
    print(f"💾 Database: {extractor.db_path}")
    print(f"{'=' * 60}")
    
    return total_facts


if __name__ == "__main__":
    extract_all_corpus()
