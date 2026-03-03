# scripts/test_extraction.py
import sys
sys.path.insert(0, '.')

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter

print("=" * 60, flush=True)
print("EXTRACTION PIPELINE TEST", flush=True)
print("=" * 60, flush=True)

# Profile
print("\n🔍 Step 1: Profiling...", flush=True)
triage = TriageAgent()
profile = triage.profile_document("corpus/tax_expenditure_ethiopia_2021_22.pdf")
print(f"  Filename: {profile.filename}", flush=True)
print(f"  Origin: {profile.origin_type} ({profile.origin_confidence:.2f})", flush=True)
print(f"  Layout: {profile.layout_complexity} ({profile.layout_confidence:.2f})", flush=True)
print(f"  Strategy: {profile.recommended_strategy}", flush=True)

# Extract
print("\n🔧 Step 2: Extracting...", flush=True)
router = ExtractionRouter()
result = router.extract("corpus/tax_expenditure_ethiopia_2021_22.pdf", profile)

print(f"  Strategy Used: {result.strategy_used}", flush=True)
print(f"  Confidence: {result.extracted_document.overall_confidence:.3f}", flush=True)
print(f"  Escalations: {result.escalation_count}", flush=True)
print(f"  Processing Time: {result.total_processing_time}s", flush=True)
print(f"  Cost: ${result.total_cost_usd:.4f}", flush=True)
print(f"  Blocks Extracted: {len(result.extracted_document.blocks)}", flush=True)
print(f"  Tables Extracted: {len(result.extracted_document.tables)}", flush=True)

# Show sample
if result.extracted_document.blocks:
    print("\n📄 Sample Block:", flush=True)
    print(f"  {result.extracted_document.blocks[0].content[:100]}...", flush=True)

if result.extracted_document.tables:
    print("\n📊 Sample Table:", flush=True)
    tbl = result.extracted_document.tables[0]
    print(f"  ID: {tbl['table_id']}", flush=True)
    print(f"  Headers: {tbl['headers'][:3]}...", flush=True)
    print(f"  Rows: {tbl['row_count']}", flush=True)

print("\n" + "=" * 60, flush=True)
print("✅ EXTRACTION PIPELINE WORKING!", flush=True)
print("=" * 60, flush=True)
