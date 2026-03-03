# tests/test_triage.py
"""
Unit Tests for Triage Agent

Tests document classification logic, threshold handling, and profile generation.
"""

import pytest
from pathlib import Path
from src.agents.triage import TriageAgent
from src.models.document_profile import DocumentProfile


class TestTriageAgent:
    """Test suite for Triage Agent classification logic."""
    
    @pytest.fixture
    def triage_agent(self):
        """Create a TriageAgent instance."""
        return TriageAgent(config_path="rubric/extraction_rules.yaml")
    
    @pytest.fixture
    def corpus_dir(self):
        """Path to corpus documents."""
        return Path("corpus")
    
    def test_agent_initialization(self, triage_agent):
        """Test that TriageAgent initializes correctly."""
        assert triage_agent is not None
        assert triage_agent.config is not None
        assert 'triage' in triage_agent.config
    
    def test_sample_pages_logic(self, triage_agent):
        """Test page sampling logic handles edge cases."""
        # Test with 161-page document (CBE Report)
        pages = triage_agent._sample_pages_to_analyze(161)
        assert 1 in pages
        assert 161 in pages  # Last page (-1)
        assert len(pages) >= 3
        
        # Test with short document
        pages = triage_agent._sample_pages_to_analyze(5)
        assert len(pages) <= 5
        assert 1 in pages
        assert 5 in pages
    
    def test_median_computation(self, triage_agent):
        """Test median calculation."""
        values = [1, 3, 5, 7, 9]
        assert triage_agent._compute_median(values) == 5.0
        
        values = [1, 2, 3, 4]
        assert triage_agent._compute_median(values) == 2.5
        
        values = []
        assert triage_agent._compute_median(values) == 0.0
    
    def test_origin_type_classification_native(self, triage_agent):
        """Test origin type classification with native digital signals."""
        # Simulate native digital metrics
        metrics = [
            {'char_density': 0.002, 'image_ratio': 0.1, 'char_count': 500},
            {'char_density': 0.003, 'image_ratio': 0.15, 'char_count': 600},
            {'char_density': 0.0025, 'image_ratio': 0.12, 'char_count': 550},
        ]
        
        origin_type, confidence = triage_agent._classify_origin_type(metrics)
        assert origin_type == "native_digital"
        assert confidence >= 0.8
    
    def test_origin_type_classification_scanned(self, triage_agent):
        """Test origin type classification with scanned image signals."""
        # Simulate scanned image metrics
        metrics = [
            {'char_density': 0.0001, 'image_ratio': 0.95, 'char_count': 5},
            {'char_density': 0.0002, 'image_ratio': 0.98, 'char_count': 10},
            {'char_density': 0.0001, 'image_ratio': 0.99, 'char_count': 0},
        ]
        
        origin_type, confidence = triage_agent._classify_origin_type(metrics)
        assert origin_type == "scanned_image"
        assert confidence >= 0.8
    
    def test_layout_complexity_table_heavy(self, triage_agent):
        """Test layout complexity classification for table-heavy documents."""
        metrics = [
            {'table_count': 3, 'image_count': 1, 'image_ratio': 0.1},
            {'table_count': 4, 'image_count': 0, 'image_ratio': 0.05},
            {'table_count': 5, 'image_count': 2, 'image_ratio': 0.15},
        ]
        
        complexity, confidence = triage_agent._classify_layout_complexity(metrics)
        assert complexity == "table_heavy"
    
    def test_profile_document_cbe(self, triage_agent, corpus_dir):
        """Test full document profiling on CBE Annual Report."""
        pdf_path = corpus_dir / "CBE ANNUAL REPORT 2023-24.pdf"
        
        if not pdf_path.exists():
            pytest.skip("CBE Report not found in corpus/")
        
        profile = triage_agent.profile_document(str(pdf_path))
        
        # Validate profile structure
        assert isinstance(profile, DocumentProfile)
        assert profile.doc_id is not None
        assert profile.filename == "CBE ANNUAL REPORT 2023-24.pdf"
        assert profile.page_count > 0
        assert profile.origin_type in ["native_digital", "scanned_image", "mixed"]
        assert profile.recommended_strategy in ["fast_text", "layout_aware", "vision_augmented"]
        
        # Validate metrics exist
        assert 'median_char_density' in profile.metrics
        assert 'median_image_ratio' in profile.metrics
        assert 'table_count' in profile.metrics
    
    def test_profile_document_audit(self, triage_agent, corpus_dir):
        """Test full document profiling on Audit Report (scanned)."""
        pdf_path = corpus_dir / "Audit Report - 2023.pdf"
        
        if not pdf_path.exists():
            pytest.skip("Audit Report not found in corpus/")
        
        profile = triage_agent.profile_document(str(pdf_path))
        
        # Validate profile structure
        assert isinstance(profile, DocumentProfile)
        assert profile.filename == "Audit Report - 2023.pdf"
        
        # Scanned documents should have low char density
        assert profile.metrics['median_char_density'] < 0.001
    
    def test_save_profile(self, triage_agent, corpus_dir, tmp_path):
        """Test profile saving to JSON."""
        pdf_path = corpus_dir / "CBE ANNUAL REPORT 2023-24.pdf"
        
        if not pdf_path.exists():
            pytest.skip("CBE Report not found in corpus/")
        
        profile = triage_agent.profile_document(str(pdf_path))
        profile_path = triage_agent.save_profile(profile, str(tmp_path))
        
        assert profile_path.exists()
        assert profile_path.suffix == ".json"
        
        # Verify JSON is valid
        import json
        with open(profile_path) as f:
            data = json.load(f)
        
        assert data['doc_id'] == profile.doc_id
        assert data['filename'] == profile.filename
    
    def test_domain_classification_financial(self, triage_agent, corpus_dir):
        """Test domain classification detects financial documents."""
        pdf_path = corpus_dir / "CBE ANNUAL REPORT 2023-24.pdf"
        
        if not pdf_path.exists():
            pytest.skip("CBE Report not found in corpus/")
        
        domain, keywords = triage_agent._classify_domain(pdf_path)
        
        # CBE should be classified as financial
        assert domain == "financial"
        assert len(keywords) > 0  # Should find some financial keywords


class TestDocumentProfile:
    """Test DocumentProfile Pydantic model validation."""
    
    def test_profile_validation(self):
        """Test that DocumentProfile validates correctly."""
        from src.models.document_profile import DocumentProfile
        
        profile = DocumentProfile(
            doc_id="test_123",
            filename="test.pdf",
            origin_type="native_digital",
            origin_confidence=0.95,
            layout_complexity="single_column",
            layout_confidence=0.90,
            page_count=100,
            recommended_strategy="fast_text",
            estimated_extraction_cost="fast_text_sufficient"
        )
        
        assert profile.doc_id == "test_123"
        assert profile.origin_confidence >= 0.0
        assert profile.origin_confidence <= 1.0
    
    def test_profile_invalid_confidence(self):
        """Test that invalid confidence values are rejected."""
        from src.models.document_profile import DocumentProfile
        
        with pytest.raises(Exception):
            DocumentProfile(
                doc_id="test_123",
                filename="test.pdf",
                origin_type="native_digital",
                origin_confidence=1.5,  # Invalid: > 1.0
                layout_complexity="single_column",
                layout_confidence=0.90,
                page_count=100,
                recommended_strategy="fast_text",
                estimated_extraction_cost="fast_text_sufficient"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])