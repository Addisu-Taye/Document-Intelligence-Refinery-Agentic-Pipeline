# scripts/test_all_corpus.py
"""
Test all 4 corpus documents through the full pipeline.
Generates artifacts and verifies each document is processed correctly.
"""

import json
from pathlib import Path
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import SemanticChunker
from src.agents.indexer import PageIndexBuilder

CORPUS_DIR = Path("corpus")
REFINERY_DIR = Path(".refinery")

DOCUMENTS = [
    "CBE ANNUAL REPORT 2023-24.pdf",
    "Audit Report - 2023.pdf",
    "fta_performance_survey_final_report_2022.pdf",
    "tax_expenditure_ethiopia_2021_22.pdf"
]

def test_document(pdf_name: str):
    """Test a single document through the full pipeline."""
    pdf_path = CORPUS_DIR / pdf_name
    
    if not pdf_path.exists():
        print(f"  ❌ {pdf_name}: NOT FOUND")
        return False
    
    print(f"\n📄 Testing: {pdf_name}")
    print("-" * 60)
    
    try:
        # Stage 1: Triage
        print("  1. Triage Agent...")
        triage = TriageAgent()
        profile = triage.profile_document(str(pdf_path))
        print(f"     Origin: {profile.origin_type} ({profile.origin_confidence:.2f})")
        print(f"     Layout: {profile.layout_complexity} ({profile.layout_confidence:.2f})")
        print(f"     Strategy: {profile.recommended_strategy}")
        
        # Stage 2: Extraction
        print("  2. Extraction Router...")
        router = ExtractionRouter()
        extraction_result = router.extract(str(pdf_path), profile)
        print(f"     Strategy Used: {extraction_result.strategy_used}")
        print(f"     Confidence: {extraction_result.extracted_document.overall_confidence:.3f}")
        print(f"     Blocks: {len(extraction_result.extracted_document.blocks)}")
        print(f"     Tables: {len(extraction_result.extracted_document.tables)}")
        
        # Stage 3: Chunking
        print("  3. Semantic Chunker...")
        chunker = SemanticChunker()
        chunk_result = chunker.chunk(extraction_result.extracted_document)
        print(f"     LDUs Created: {chunk_result.total_ldus}")
        print(f"     Validation: {'✓ PASS' if chunk_result.validation_passed else '✗ FAIL'}")
        
        # Stage 4: PageIndex
        print("  4. PageIndex Builder...")
        indexer = PageIndexBuilder()
        pageindex = indexer.build(
            ldus=chunk_result.ldus,
            doc_id=profile.doc_id,
            filename=pdf_name
        )
        print(f"     Nodes Created: {pageindex.total_nodes}")
        print(f"     Max Depth: {pageindex.max_depth}")
        
        # Save artifacts
        print("  5. Saving Artifacts...")
        
        # Save PageIndex
        pageindex_path = REFINERY_DIR / "pageindex" / f"{profile.doc_id}_pageindex.json"
        pageindex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pageindex_path, 'w', encoding='utf-8') as f:
            f.write(pageindex.model_dump_json(indent=2))
        print(f"     ✓ {pageindex_path}")
        
        print(f"\n  ✅ {pdf_name}: COMPLETE")
        return True
        
    except Exception as e:
        print(f"\n  ❌ {pdf_name}: FAILED - {str(e)}")
        return False

def main():
    """Test all corpus documents."""
    print("=" * 60)
    print("TRP1 WEEK 3: FULL CORPUS TEST")
    print("=" * 60)
    
    REFINERY_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    for doc in DOCUMENTS:
        result = test_document(doc)
        results.append((doc, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for doc, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {doc}")
    
    print(f"\nTotal: {passed}/{total} documents processed successfully")
    
    if passed == total:
        print("\n🎉 ALL DOCUMENTS TESTED SUCCESSFULLY!")
    else:
        print(f"\n⚠️ {total - passed} document(s) failed - review errors above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
