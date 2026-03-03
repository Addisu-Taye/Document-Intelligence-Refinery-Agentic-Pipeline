# scripts/profile_all_corpus.py
"""
Profile all 4 corpus documents and save DocumentProfiles.

This script generates the .refinery/profiles/*.json files
required for Interim Submission.
"""

from pathlib import Path
from src.agents.triage import TriageAgent


def main():
    corpus_dir = Path("corpus")
    output_dir = Path(".refinery/profiles")
    
    # List of corpus documents
    documents = [
        "CBE ANNUAL REPORT 2023-24.pdf",
        "Audit Report - 2023.pdf",
        "fta_performance_survey_final_report_2022.pdf",
        "tax_expenditure_ethiopia_2021_22.pdf"
    ]
    
    agent = TriageAgent()
    
    print("🔍 Profiling all corpus documents...\n")
    print("=" * 60)
    
    for doc_name in documents:
        pdf_path = corpus_dir / doc_name
        
        if not pdf_path.exists():
            print(f"❌ Not found: {doc_name}")
            continue
        
        print(f"\n📄 Processing: {doc_name}")
        print("-" * 60)
        
        try:
            profile = agent.profile_document(str(pdf_path))
            profile_path = agent.save_profile(profile, str(output_dir))
            
            print(f"  Origin Type: {profile.origin_type} ({profile.origin_confidence:.2f})")
            print(f"  Layout: {profile.layout_complexity} ({profile.layout_confidence:.2f})")
            print(f"  Domain: {profile.domain_hint}")
            print(f"  Pages: {profile.page_count}")
            print(f"  Strategy: {profile.recommended_strategy}")
            print(f"  Cost: {profile.estimated_extraction_cost}")
            print(f"  Escalation: {profile.requires_escalation}")
            print(f"  Saved: {profile_path}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Corpus profiling complete!")
    print(f"📁 Profiles saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()