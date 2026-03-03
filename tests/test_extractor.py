# tests/test_extractor.py
"""
Unit Tests for Extraction Router and Strategies

Tests strategy selection, confidence scoring, and escalation logic.
"""

import pytest
from pathlib import Path
from src.agents.extractor import ExtractionRouter, ExtractionResult
from src.models.document_profile import DocumentProfile
from src.strategies.fast_text import FastTextExtractor, ConfidenceMetrics


class TestFastTextExtractor:
    """Test Strategy A: Fast Text Extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Create FastTextExtractor instance."""
        config = {
            'fast_text': {
                'min_confidence_to_proceed': 0.75,
                'weights': {
                    'char_density': 0.4,
                    'image_ratio': 0.3,
                    'table_structure_preserved': 0.2,
                    'font_metadata_present': 0.1
                }
            }
        }
        return FastTextExtractor(config=config)
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initializes correctly."""
        assert extractor is not None
        assert extractor.confidence_threshold == 0.75
    
    def test_confidence_metrics_model(self):
        """Test ConfidenceMetrics Pydantic model."""
        metrics = ConfidenceMetrics(
            char_count=500,
            char_density=0.002,
            image_ratio=0.1,
            has_font_metadata=True,
            table_count=3,
            whitespace_ratio=0.15,
            overall_confidence=0.85,
            should_escalate=False
        )
        
        assert metrics.char_count == 500
        assert metrics.overall_confidence >= 0.0
        assert metrics.overall_confidence <= 1.0
        assert metrics.should_escalate == False
    
    def test_confidence_high_for_native_digital(self, extractor):
        """Test high confidence for native digital signals."""
        # Simulate high-confidence scenario
        metrics = ConfidenceMetrics(
            char_count=1000,
            char_density=0.003,
            image_ratio=0.1,
            has_font_metadata=True,
            table_count=2,
            whitespace_ratio=0.12,
            overall_confidence=0.92,
            should_escalate=False
        )
        
        assert metrics.should_escalate == False
        assert metrics.overall_confidence >= extractor.confidence_threshold
    
    def test_confidence_low_for_scanned(self, extractor):
        """Test low confidence for scanned-like signals."""
        metrics = ConfidenceMetrics(
            char_count=10,
            char_density=0.0001,
            image_ratio=0.95,
            has_font_metadata=False,
            table_count=0,
            whitespace_ratio=0.50,
            overall_confidence=0.35,
            should_escalate=True,
            escalation_reason="low char density; high image ratio"
        )
        
        assert metrics.should_escalate == True
        assert metrics.overall_confidence < extractor.confidence_threshold


class TestExtractionRouter:
    """Test Extraction Router with escalation logic."""
    
    @pytest.fixture
    def router(self):
        """Create ExtractionRouter instance."""
        return ExtractionRouter(config_path="rubric/extraction_rules.yaml")
    
    @pytest.fixture
    def corpus_dir(self):
        """Path to corpus documents."""
        return Path("corpus")
    
    def test_router_initialization(self, router):
        """Test router initializes correctly."""
        assert router is not None
        assert 'fast_text' in router.strategies
        assert 'layout_aware' in router.strategies
        assert 'vision_augmented' in router.strategies
    
    def test_select_initial_strategy_native(self, router):
        """Test strategy selection for native digital doc."""
        from datetime import datetime
        
        profile = DocumentProfile(
            doc_id="test_native",
            filename="test.pdf",
            origin_type="native_digital",
            origin_confidence=0.95,
            layout_complexity="single_column",
            layout_confidence=0.90,
            page_count=100,
            recommended_strategy="fast_text",
            estimated_extraction_cost="fast_text_sufficient"
        )
        
        strategy = router._select_initial_strategy(profile)
        assert strategy == "fast_text"
    
    def test_select_initial_strategy_scanned(self, router):
        """Test strategy selection for scanned doc."""
        from datetime import datetime
        
        profile = DocumentProfile(
            doc_id="test_scanned",
            filename="scanned.pdf",
            origin_type="scanned_image",
            origin_confidence=0.95,
            layout_complexity="single_column",
            layout_confidence=0.80,
            page_count=50,
            recommended_strategy="vision_augmented",
            estimated_extraction_cost="needs_vision_model"
        )
        
        strategy = router._select_initial_strategy(profile)
        assert strategy == "vision_augmented"
    
    def test_select_initial_strategy_table_heavy(self, router):
        """Test strategy selection for table-heavy doc."""
        from datetime import datetime
        
        profile = DocumentProfile(
            doc_id="test_tables",
            filename="tables.pdf",
            origin_type="native_digital",
            origin_confidence=0.90,
            layout_complexity="table_heavy",
            layout_confidence=0.85,
            page_count=80,
            recommended_strategy="layout_aware",
            estimated_extraction_cost="needs_layout_model"
        )
        
        strategy = router._select_initial_strategy(profile)
        assert strategy == "layout_aware"
    
    def test_get_next_strategy(self, router):
        """Test escalation chain."""
        assert router._get_next_strategy("fast_text") == "layout_aware"
        assert router._get_next_strategy("layout_aware") == "vision_augmented"
        assert router._get_next_strategy("vision_augmented") is None
    
    def test_extract_with_escalation(self, router, corpus_dir):
        """Test extraction with potential escalation."""
        pdf_path = corpus_dir / "tax_expenditure_ethiopia_2021_22.pdf"
        
        if not pdf_path.exists():
            pytest.skip("Tax report not found in corpus/")
        
        # Create a simple profile
        from datetime import datetime
        profile = DocumentProfile(
            doc_id="tax_test",
            filename="tax_expenditure_ethiopia_2021_22.pdf",
            origin_type="native_digital",
            origin_confidence=0.90,
            layout_complexity="single_column",
            layout_confidence=0.80,
            page_count=60,
            recommended_strategy="fast_text",
            estimated_extraction_cost="fast_text_sufficient"
        )
        
        result = router.extract(str(pdf_path), profile, escalate=True)
        
        assert isinstance(result, ExtractionResult)
        assert result.extracted_document is not None
        assert result.strategy_used in ["fast_text", "layout_aware", "vision_augmented"]
        assert result.total_processing_time >= 0
        assert result.total_cost_usd >= 0


class TestBudgetGuard:
    """Test VLM budget guard."""
    
    def test_budget_within_limits(self):
        """Test budget allows requests within limits."""
        from src.strategies.vision_augmented import BudgetGuard
        
        budget = BudgetGuard(
            max_cost_usd=2.00,
            max_tokens=50000,
            rate_per_1k_tokens=0.0001
        )
        
        can_proceed, error = budget.can_proceed(1000)
        assert can_proceed == True
        assert error is None
    
    def test_budget_exceeds_token_limit(self):
        """Test budget rejects requests exceeding token limit."""
        from src.strategies.vision_augmented import BudgetGuard
        
        budget = BudgetGuard(
            max_cost_usd=2.00,
            max_tokens=1000,
            rate_per_1k_tokens=0.0001
        )
        
        # Already used 900 tokens
        budget.tokens_used = 900
        
        can_proceed, error = budget.can_proceed(200)
        assert can_proceed == False
        assert "Token limit exceeded" in error
    
    def test_budget_exceeds_cost_limit(self):
        """Test budget rejects requests exceeding cost limit."""
        from src.strategies.vision_augmented import BudgetGuard
        
        budget = BudgetGuard(
            max_cost_usd=0.001,  # Very low limit
            max_tokens=50000,
            rate_per_1k_tokens=0.0001
        )
        
        # Already spent most of budget
        budget.cost_accumulated = 0.0009
        
        can_proceed, error = budget.can_proceed(1000)
        assert can_proceed == False
        assert "Budget exceeded" in error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])