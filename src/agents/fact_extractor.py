# src/agents/fact_extractor.py — UNIVERSAL VERSION (No Model Imports)
"""
FactTable Extractor — Universal version without model dependencies.
Works with any document structure.
"""

import sqlite3
import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('.refinery/fact_extractor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FactTableExtractor:
    """Extract structured facts from documents into SQLite FactTable."""
    
    ENTITIES = {
        'net_profit': ['net profit', 'profit', 'net income', 'earnings'],
        'revenue': ['revenue', 'income', 'sales', 'turnover'],
        'expense': ['expense', 'expenses', 'expenditure', 'cost'],
        'assets': ['assets', 'total assets'],
        'equity': ['equity', "shareholders' equity", 'capital'],
        'liability': ['liability', 'liabilities', 'debt'],
        'cash': ['cash', 'cash flow'],
        'dividend': ['dividend', 'dividends'],
    }
    
    def __init__(self, db_path: str = ".refinery/facts.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite FactTable."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT NOT NULL,
                value TEXT,
                amount TEXT,
                figure TEXT,
                metric_value TEXT,
                doc_id TEXT,
                filename TEXT,
                page_refs TEXT,
                content_hash TEXT,
                extraction_confidence REAL,
                fact_id TEXT UNIQUE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity ON facts(entity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_id ON facts(doc_id)')
        
        conn.commit()
        conn.close()
        logger.info(f"✓ FactTable initialized: {self.db_path}")
    
    def extract_from_document(self, doc: Any, doc_id: str, filename: str) -> List[Dict]:
        """Extract facts from document using duck-typing (no model imports)."""
        facts = []
        
        # Try to get blocks/tables using getattr (works with any object)
        blocks = getattr(doc, 'blocks', []) or []
        tables = getattr(doc, 'tables', []) or []
        
        # Extract from blocks
        for block in blocks:
            if block:
                content = getattr(block, 'content', None) or getattr(block, 'text', None)
                page_refs = getattr(block, 'page_refs', None)
                if content:
                    block_facts = self._extract_from_text(str(content), doc_id, filename, page_refs)
                    facts.extend(block_facts)
        
        # Extract from tables
        for table in tables:
            if table:
                table_facts = self._extract_from_table(table, doc_id, filename)
                facts.extend(table_facts)
        
        self._store_facts(facts)
        logger.info(f"✓ Extracted {len(facts)} facts from {filename}")
        return facts
    
    def _extract_from_table(self, table: Any, doc_id: str, filename: str) -> List[Dict]:
        """Extract facts from table using duck-typing."""
        facts = []
        
        headers = getattr(table, 'headers', None) or getattr(table, 'cols', []) or []
        rows = getattr(table, 'rows', None) or getattr(table, 'data', []) or []
        page_refs = getattr(table, 'page_refs', None)
        
        if not headers or not rows:
            return facts
        
        # Find entity columns
        entity_cols = {}
        for i, header in enumerate(headers):
            if header:
                header_lower = str(header).lower()
                for entity, keywords in self.ENTITIES.items():
                    if any(kw in header_lower for kw in keywords):
                        entity_cols[i] = entity
                        break
        
        # Extract from rows
        for row_idx, row in enumerate(rows[:20]):
            if isinstance(row, (list, tuple)):
                for col_idx, entity in entity_cols.items():
                    if col_idx < len(row):
                        cell = row[col_idx]
                        if cell and self._is_financial_value(str(cell)):
                            fact = self._create_fact(
                                entity=entity,
                                value=str(cell),
                                doc_id=doc_id,
                                filename=filename,
                                page_refs=page_refs,
                                content_hash=f"table_r{row_idx}_c{col_idx}",
                                confidence=0.85
                            )
                            facts.append(fact)
        
        return facts
    
    def _extract_from_text(self, text: str, doc_id: str, filename: str, page_refs: Any) -> List[Dict]:
        """Extract facts from text using regex."""
        facts = []
        text_lower = text.lower()
        
        for entity, keywords in self.ENTITIES.items():
            for keyword in keywords:
                if keyword in text_lower:
                    values = self._extract_financial_values(text, keyword)
                    for value in values:
                        fact = self._create_fact(
                            entity=entity,
                            value=value,
                            doc_id=doc_id,
                            filename=filename,
                            page_refs=page_refs,
                            content_hash=f"text_{hash(text) % 10000}",
                            confidence=0.75
                        )
                        facts.append(fact)
                    break
        
        return facts
    
    def _extract_financial_values(self, text: str, keyword: str) -> List[str]:
        """Extract financial values near keyword."""
        values = []
        pos = text.lower().find(keyword)
        if pos == -1:
            return values
        
        start = max(0, pos - 100)
        end = min(len(text), pos + 100)
        window = text[start:end]
        
        patterns = [
            r'(?:ETB|Birr|\$|USD|€)?\s*([\d,]+\.?\d*)\s*(billion|million|thousand)',
            r'([\d,]+\.?\d*)\s*(billion|million|thousand)\s*(?:ETB|Birr|\$|USD|€)?',
            r'([\d]{7,})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, window, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    value = " ".join([g for g in match if g])
                else:
                    value = match
                if value and value not in values:
                    values.append(value)
        
        return values
    
    def _is_financial_value(self, text: str) -> bool:
        """Check if text is a financial value."""
        if not text:
            return False
        if re.search(r'[\d,]+\.?\d*\s*(billion|million|thousand|%|\$|ETB|Birr)', text, re.I):
            return True
        if re.search(r'[\d]{7,}', text):
            return True
        return False
    
    def _create_fact(self, entity: str, value: str, doc_id: str, filename: str, 
                    page_refs: Any, content_hash: str, confidence: float) -> Dict:
        """Create fact with value in multiple field names."""
        return {
            "entity": entity,
            "value": value,
            "amount": value,
            "figure": value,
            "metric_value": value,
            "doc_id": doc_id,
            "filename": filename,
            "page_refs": json.dumps(page_refs) if page_refs else '[]',
            "content_hash": content_hash,
            "extraction_confidence": confidence,
            "fact_id": f"{doc_id}_{entity}_{content_hash[:8]}"
        }
    
    def _store_facts(self, facts: List[Dict]):
        """Store facts in SQLite."""
        if not facts:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for fact in facts:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO facts 
                    (entity, value, amount, figure, metric_value, 
                     doc_id, filename, page_refs, content_hash, 
                     extraction_confidence, fact_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    fact.get('entity'),
                    fact.get('value'),
                    fact.get('amount'),
                    fact.get('figure'),
                    fact.get('metric_value'),
                    fact.get('doc_id'),
                    fact.get('filename'),
                    fact.get('page_refs'),
                    fact.get('content_hash'),
                    fact.get('extraction_confidence'),
                    fact.get('fact_id'),
                    fact.get('created_at', datetime.now(timezone.utc).isoformat())
                ))
            except Exception as e:
                logger.warning(f"Failed to store fact: {e}")
        
        conn.commit()
        conn.close()
    
    def query_facts(self, entity: Optional[str] = None, doc_id: Optional[str] = None) -> List[Dict]:
        """Query facts with flexible value field lookup."""
        conn = sqlite3.connect(self.db_path)
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
        
        query += " ORDER BY extraction_confidence DESC LIMIT 20"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        facts = []
        for row in rows:
            fact = dict(row)
            if not fact.get('value'):
                fact['value'] = (
                    fact.get('amount')
                    or fact.get('figure')
                    or fact.get('metric_value')
                    or ''
                )
            facts.append(fact)
        
        return facts
    
    def get_stats(self) -> Dict:
        """Get FactTable statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM facts")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT entity) FROM facts")
        entities = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM facts")
        docs = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_facts": total,
            "unique_entities": entities,
            "documents": docs
        }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    """Re-extract facts from all processed documents."""
    import sys
    sys.path.insert(0, '.')
    
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from pathlib import Path
    
    print("🔄 Re-extracting facts from all documents...")
    
    extractor = FactTableExtractor()
    triage = TriageAgent()
    router = ExtractionRouter()
    
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
            profile = triage.profile_document(str(pdf_path))
            extraction = router.extract(str(pdf_path), profile)
            facts = extractor.extract_from_document(extraction.extracted_document, doc_id, filename)
            print(f"   ✓ Extracted {len(facts)} facts")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    stats = extractor.get_stats()
    print(f"\n✅ FactTable stats: {stats['total_facts']} facts, {stats['unique_entities']} entities, {stats['documents']} docs")