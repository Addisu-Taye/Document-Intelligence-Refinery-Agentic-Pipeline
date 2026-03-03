# src/agents/extractor.py
"""
Extraction Router with Escalation Guard

Routes documents to appropriate extraction strategy based on DocumentProfile.
Implements confidence-gated escalation: Strategy A → B → C.

This is Deliverable #4 from the challenge specification.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_aware import LayoutExtractor
from src.strategies.vision_augmented import VisionExtractor
from src.agents.config import Config


class EscalationLogEntry(BaseModel):
    """Log entry for strategy escalation."""
    from_strategy: Literal["fast_text", "layout_aware", "vision_augmented"]
    to_strategy: Literal["fast_text", "layout_aware", "vision_augmented"]
    reason: str
    confidence_score: float
    threshold: float
    pages: list[int] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExtractionResult(BaseModel):
    """Wrapped extraction result with metadata."""
    extracted_document: ExtractedDocument
    strategy_used: Literal["fast_text", "layout_aware", "vision_augmented"]
    escalation_count: int
    escalation_log: list[EscalationLogEntry]
    total_processing_time: float
    total_cost_usd: float


class ExtractionRouter:
    """
    Routes documents to appropriate extraction strategy with escalation guard.
    
    Pattern:
    1. Start with Strategy A (fast_text) for native digital docs
    2. If confidence < threshold, escalate to Strategy B (layout_aware)
    3. If confidence < threshold, escalate to Strategy C (vision_augmented)
    4. Strategy C is final - no further escalation
    
    This prevents "garbage in, hallucination out" RAG failures.
    """
    
    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        """
        Initialize router with configuration.
        
        Args:
            config_path: Path to extraction_rules.yaml
        """
        self.config = Config.load(config_path)
        
        # Initialize strategies with config
        self.strategies = {
            'fast_text': FastTextExtractor(config=self.config.get('confidence_scoring', {})),
            'layout_aware': LayoutExtractor(config=self.config.get('confidence_scoring', {})),
            'vision_augmented': VisionExtractor(config=self.config)
        }
        
        # Escalation configuration
        self.escalation_config = self.config.get('escalation', {})
        self.max_escalations = self.escalation_config.get('max_escalations_per_doc', 2)
        
        # Strategy order
        self.strategy_order = ['fast_text', 'layout_aware', 'vision_augmented']
    
    def _select_initial_strategy(self, doc_profile: DocumentProfile) -> Literal["fast_text", "layout_aware", "vision_augmented"]:
        """
        Select initial extraction strategy based on DocumentProfile.
        
        Args:
            doc_profile: Document profile from Triage Agent
        
        Returns:
            Strategy name to start with
        """
        # Follow profile recommendation
        recommended = doc_profile.recommended_strategy
        
        # Override based on origin type
        if doc_profile.origin_type == "scanned_image":
            return "vision_augmented"
        elif doc_profile.layout_complexity in ["table_heavy", "multi_column"]:
            return "layout_aware"
        elif doc_profile.origin_type == "native_digital" and doc_profile.layout_complexity == "single_column":
            return "fast_text"
        
        return recommended
    
    def _get_next_strategy(self, current_strategy: str) -> Optional[str]:
        """Get next strategy in escalation chain."""
        current_idx = self.strategy_order.index(current_strategy)
        
        if current_idx >= len(self.strategy_order) - 1:
            return None  # Already at final strategy
        
        return self.strategy_order[current_idx + 1]
    
    def _log_to_ledger(self, result: ExtractionResult, output_path: str = ".refinery/extraction_ledger.jsonl"):
        """Log extraction result to ledger."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        ledger_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "doc_id": result.extracted_document.doc_id,
            "filename": result.extracted_document.filename,
            "page_count": result.extracted_document.pages_processed,
            "extraction_strategy": result.strategy_used,
            "overall_confidence": result.extracted_document.overall_confidence,
            "escalation_count": result.escalation_count,
            "escalation_log": [log.model_dump(mode='json') for log in result.escalation_log],
            "processing_time_seconds": result.total_processing_time,
            "cost_estimate_usd": result.total_cost_usd,
            "processing_status": "completed",
            "extraction_status": "completed"
        }
        
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(ledger_entry, default=str) + "\n")
    
    def extract(self, pdf_path: str, doc_profile: DocumentProfile, escalate: bool = True) -> ExtractionResult:
        """
        Main extraction method with escalation guard.
        
        Args:
            pdf_path: Path to PDF file
            doc_profile: DocumentProfile from Triage Agent
            escalate: If True, automatically escalate on low confidence
        
        Returns:
            ExtractionResult with extracted document and metadata
        """
        start_time = time.time()
        escalation_count = 0
        escalation_log = []
        
        # Select initial strategy
        current_strategy = self._select_initial_strategy(doc_profile)
        
        while True:
            # Get strategy instance
            strategy = self.strategies[current_strategy]
            
            # Extract
            extracted_doc = strategy.extract(pdf_path, doc_profile)
            
            # Check if escalation is needed
            should_escalate_flag, escalation_reason = strategy.should_escalate(extracted_doc)
            
            if should_escalate_flag and escalate and escalation_count < self.max_escalations:
                # Get next strategy
                next_strategy = self._get_next_strategy(current_strategy)
                
                if next_strategy:
                    # Log escalation
                    escalation_entry = EscalationLogEntry(
                        from_strategy=current_strategy,
                        to_strategy=next_strategy,
                        reason=escalation_reason or "confidence below threshold",
                        confidence_score=extracted_doc.overall_confidence,
                        threshold=self.config.get('confidence_scoring', {}).get(current_strategy, {}).get('min_confidence_to_proceed', 0.75)
                    )
                    escalation_log.append(escalation_entry)
                    escalation_count += 1
                    
                    # Continue with next strategy
                    current_strategy = next_strategy
                    continue
            
            # No escalation needed or max escalations reached
            break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Create result
        result = ExtractionResult(
            extracted_document=extracted_doc,
            strategy_used=current_strategy,
            escalation_count=escalation_count,
            escalation_log=escalation_log,
            total_processing_time=round(total_time, 2),
            total_cost_usd=extracted_doc.cost_estimate_usd
        )
        
        # Log to ledger
        self._log_to_ledger(result)
        
        return result
    
    def extract_all(self, pdf_paths: list[str], profiles: list[DocumentProfile]) -> list[ExtractionResult]:
        """
        Extract multiple documents.
        
        Args:
            pdf_paths: List of PDF paths
            profiles: Corresponding DocumentProfiles
        
        Returns:
            List of ExtractionResults
        """
        results = []
        
        for pdf_path, profile in zip(pdf_paths, profiles):
            result = self.extract(pdf_path, profile)
            results.append(result)
        
        return results


def main():
    """CLI entry point for extraction."""
    import sys
    from src.agents.triage import TriageAgent
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.extractor <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # First, profile the document
    triage = TriageAgent()
    profile = triage.profile_document(pdf_path)
    
    print(f"📄 Document Profile:")
    print(f"  Origin: {profile.origin_type} ({profile.origin_confidence:.2f})")
    print(f"  Layout: {profile.layout_complexity} ({profile.layout_confidence:.2f})")
    print(f"  Strategy: {profile.recommended_strategy}")
    
    # Extract with router
    router = ExtractionRouter()
    result = router.extract(pdf_path, profile)
    
    print(f"\n🔧 Extraction Result:")
    print(f"  Strategy Used: {result.strategy_used}")
    print(f"  Confidence: {result.extracted_document.overall_confidence:.3f}")
    print(f"  Escalations: {result.escalation_count}")
    print(f"  Processing Time: {result.total_processing_time}s")
    print(f"  Cost: ${result.total_cost_usd:.4f}")
    
    if result.escalation_log:
        print(f"\n⬆️  Escalation Log:")
        for entry in result.escalation_log:
            print(f"  {entry.from_strategy} → {entry.to_strategy}: {entry.reason}")


if __name__ == "__main__":
    main()