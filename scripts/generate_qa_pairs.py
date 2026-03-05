# scripts/generate_qa_pairs.py
"""
Generate 12 Q&A Pairs (3 per document) with ProvenanceChain citations.
Final Submission Requirement: Artifacts
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from src.agents.query_agent import QueryAgent

OUTPUT_DIR = Path(".refinery/qa_pairs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define 3 questions per document based on content
QA_PAIRS = {
    "cbe_annual_report_2023_24": [
        {
            "question": "What was the net profit for FY 2023-24?",
            "expected_topic": "net profit, financial performance"
        },
        {
            "question": "What are the total assets of the bank?",
            "expected_topic": "assets, balance sheet"
        },
        {
            "question": "What is the bank's capital adequacy ratio?",
            "expected_topic": "capital adequacy, regulatory compliance"
        }
    ],
    "audit_report_-_2023": [
        {
            "question": "What is the auditor's opinion on the financial statements?",
            "expected_topic": "auditor opinion, financial statements"
        },
        {
            "question": "Were there any material misstatements found?",
            "expected_topic": "material misstatement, audit findings"
        },
        {
            "question": "What is the reporting period covered?",
            "expected_topic": "reporting period, fiscal year"
        }
    ],
    "fta_performance_survey_final_rep": [
        {
            "question": "What are the key findings on financial transparency?",
            "expected_topic": "financial transparency, key findings"
        },
        {
            "question": "What recommendations were made for improvement?",
            "expected_topic": "recommendations, improvement"
        },
        {
            "question": "Which ministries were assessed in the survey?",
            "expected_topic": "ministries, assessment scope"
        }
    ],
    "tax_expenditure_ethiopia_2021_22": [
        {
            "question": "What is the total tax expenditure for the reporting period?",
            "expected_topic": "tax expenditure, total amount"
        },
        {
            "question": "Which sectors received the largest tax incentives?",
            "expected_topic": "tax incentives, sectors"
        },
        {
            "question": "What is the revenue foregone from import tax exemptions?",
            "expected_topic": "revenue foregone, import tax"
        }
    ]
}

def generate_qa_pairs():
    """Generate Q&A pairs with ProvenanceChain for all 4 documents."""
    agent = QueryAgent()
    
    print("=" * 60)
    print("GENERATING Q&A PAIRS (Final Submission Requirement)")
    print("=" * 60)
    
    all_pairs = []
    
    for doc_id, questions in QA_PAIRS.items():
        print(f"\n📄 {doc_id}...")
        
        doc_pairs = []
        for i, qa in enumerate(questions, 1):
            print(f"  Q{i}: {qa['question']}")
            
            # Query the agent
            result = agent.answer(
                query=qa['question'],
                doc_id=doc_id,
                ldu_store={}  # Would load from storage in production
            )
            
            # Build Q&A pair with ProvenanceChain
            pair = {
                "document_id": doc_id,
                "question_number": i,
                "question": qa['question'],
                "answer": result.answer,
                "confidence": result.answer_confidence,
                "retrieval_method": result.retrieval_method,
                "verification_status": result.verification_status,
                "citations": [
                    {
                        "document_name": c.document_name,
                        "page_number": c.page_number,
                        "bounding_box": c.bounding_box,
                        "content_hash": c.content_hash,
                        "cited_text": c.cited_text[:200],
                        "extraction_strategy": c.extraction_strategy,
                        "extraction_confidence": c.extraction_confidence
                    }
                    for c in result.citations
                ],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            doc_pairs.append(pair)
            print(f"    ✓ Answer generated (confidence: {result.answer_confidence:.2f})")
            print(f"    ✓ Citations: {len(result.citations)}")
        
        # Save per-document Q&A file
        output_path = OUTPUT_DIR / f"{doc_id}_qa_pairs.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_pairs, f, indent=2, default=str)
        
        print(f"  ✓ Saved: {output_path.name}")
        all_pairs.extend(doc_pairs)
    
    # Save combined file
    combined_path = OUTPUT_DIR / "all_qa_pairs.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, indent=2, default=str)
    
    print(f"\n{'=' * 60}")
    print(f"✅ Generated {len(all_pairs)} Q&A pairs (3 per document)")
    print(f"📍 Location: {OUTPUT_DIR.absolute()}")
    print(f"{'=' * 60}")
    
    return len(all_pairs)

if __name__ == "__main__":
    generate_qa_pairs()
