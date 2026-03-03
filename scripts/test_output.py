# scripts/test_output.py
import sys
import os

print("=" * 50, flush=True)
print("PYTHON OUTPUT TEST", flush=True)
print("=" * 50, flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"Current directory: {os.getcwd()}", flush=True)
print(f"Script file: {__file__}", flush=True)

# Test imports
try:
    from src.agents.triage import TriageAgent
    print("✓ TriageAgent import: SUCCESS", flush=True)
except Exception as e:
    print(f"✗ TriageAgent import: FAILED - {e}", flush=True)

# Test profiling
try:
    print("\n🔍 Testing profiling...", flush=True)
    triage = TriageAgent()
    profile = triage.profile_document("corpus/tax_expenditure_ethiopia_2021_22.pdf")
    print(f"✓ Filename: {profile.filename}", flush=True)
    print(f"✓ Origin: {profile.origin_type}", flush=True)
    print(f"✓ Strategy: {profile.recommended_strategy}", flush=True)
except Exception as e:
    print(f"✗ Profiling failed: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("=" * 50, flush=True)
print("TEST COMPLETE", flush=True)
print("=" * 50, flush=True)
