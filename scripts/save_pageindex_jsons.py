# scripts/save_pageindex_jsons.py
"""Save PageIndex JSON files for all 4 corpus documents (Final Submission Requirement)."""

import json
from pathlib import Path
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import SemanticChunker
from src.agents.indexer import PageIndexBuilder

CORPUS_DIR = Path("corpus")
OUTPUT_DIR = Path(".refinery/pageindex")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENTS = [
    "CBE ANNUAL REPORT 2023-24.pdf",
    "Audit Report - 2023.pdf",
    "fta_performance_survey_final_report_2022.pdf",
    "tax_expenditure_ethiopia_2021_22.pdf"
]

print("=" * 60)
print("SAVING PAGEINDEX JSONS (Final Submission Requirement)")
print("=" * 60)

triage = TriageAgent()
router = ExtractionRouter()
chunker = SemanticChunker()
indexer = PageIndexBuilder()

saved = 0
for doc_name in DOCUMENTS:
    doc_path = CORPUS_DIR / doc_name
    if not doc_path.exists():
        print(f"\n⚠️  {doc_name}: NOT FOUND")
        continue
    
    print(f"\n📄 {doc_name}...")
    
    try:
        doc_id = doc_path.stem.replace(" ", "_").lower()[:32]
        
        # Full pipeline: Profile → Extract → Chunk → Index
        profile = triage.profile_document(str(doc_path))
        extraction = router.extract(str(doc_path), profile)
        chunks = chunker.chunk(extraction.extracted_document)
        pageindex = indexer.build(chunks.ldus, doc_id, doc_name)
        
        # Save JSON
        output_path = OUTPUT_DIR / f"{doc_id}_pageindex.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pageindex.model_dump_json(indent=2))
        
        print(f"  ✓ Saved: {output_path.name}")
        print(f"    Nodes: {pageindex.total_nodes}, Max Depth: {pageindex.max_depth}")
        saved += 1
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'=' * 60}")
print(f"✅ Saved {saved}/4 PageIndex JSON files")
print(f"📍 Location: {OUTPUT_DIR.absolute()}")
print(f"{'=' * 60}")
