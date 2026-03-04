# tests/test_query_agent.py
"""
Unit Tests for Query Agent

Tests query parsing, PageIndex search, answer synthesis, and citation building.
"""

import pytest
from src.agents.query_agent import QueryAgent, QueryState
from src.models.provenance import ProvenanceChain, ProvenanceCitation


class TestQueryAgent:
    """Test QueryAgent functionality."""
    
    @pytest.fixture
    def agent(self):
        """Create QueryAgent instance."""
        return QueryAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.config is not None
    
    def test_extract_query_keywords(self, agent):
        """Test keyword extraction from query."""
        query = "What was the net profit of Commercial Bank of Ethiopia in 2024?"
        keywords = agent._extract_query_keywords(query)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "profit" in keywords or "ethiopia" in keywords or "bank" in keywords
    
    def test_search_pageindex_stub(self, agent):
        """Test PageIndex search (stub - would require actual PageIndex)."""
        # This test would require a mock PageIndex
        # For now, verify method exists and returns list
        assert hasattr(agent, '_search_pageindex')
    
    def test_synthesize_answer_fast_text(self, agent):
        """Test answer synthesis with fast_text strategy."""
        from src.models.ldu import LDU
        
        ldus = [
            LDU(
                ldu_id="test_ldu_001",
                content="Net profit increased to ETB 14.2 billion in FY 2023-24.",
                chunk_type="text_block",
                page_refs=[10],
                token_count=20,
                word_count=12,
                content_hash="hash_001",
                extraction_strategy="fast_text",
                extraction_confidence=0.88
            )
        ]
        
        answer, confidence = agent._synthesize_answer(
            query="What was the net profit?",
            ldus=ldus,
            strategy="fast_text"
        )
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert confidence >= 0.0 and confidence <= 1.0
        assert "14.2 billion" in answer or "profit" in answer.lower()
    
    def test_synthesize_answer_layout_aware(self, agent):
        """Test answer synthesis with layout_aware strategy for tables."""
        from src.models.ldu import LDU
        
        ldus = [
            LDU(
                ldu_id="test_tbl_001",
                content="Headers: Metric | 2023 | 2024\nData: Net Profit | 12.8B | 14.2B",
                chunk_type="table",
                page_refs=[11],
                token_count=25,
                word_count=15,
                table_headers=["Metric", "2023", "2024"],
                table_row_count=1,
                table_data=[{"Metric": "Net Profit", "2023": "12.8B", "2024": "14.2B"}],
                content_hash="hash_tbl",
                extraction_strategy="layout_aware",
                extraction_confidence=0.92
            )
        ]
        
        answer, confidence = agent._synthesize_answer(
            query="Show me the profit table",
            ldus=ldus,
            strategy="layout_aware"
        )
        
        assert isinstance(answer, str)
        assert confidence >= 0.80  # Layout-aware should have higher confidence
        assert "14.2B" in answer or "table" in answer.lower()
    
    def test_build_citations(self, agent):
        """Test citation building from LDUs."""
        from src.models.ldu import LDU, BoundingBox
        
        ldus = [
            LDU(
                ldu_id="ldu_cite_001",
                content="Important financial data here.",
                chunk_type="text_block",
                page_refs=[15],
                bounding_box=BoundingBox(x0=72, y0=400, x1=300, y1=450),
                token_count=15,
                word_count=5,
                content_hash="hash_cite",
                extraction_strategy="fast_text",
                extraction_confidence=0.85,
                section_path=["Financial Statements", "Highlights"]
            )
        ]
        
        citations = agent._build_citations(ldus, query="financial data")
        
        assert isinstance(citations, list)
        if citations:
            citation = citations[0]
            assert isinstance(citation, ProvenanceCitation)
            assert citation.page_number == 15
            assert citation.ldu_id == "ldu_cite_001"
    
    def test_answer_no_pageindex(self, agent):
        """Test answer when PageIndex not found."""
        result = agent.answer(
            query="Test query",
            doc_id="nonexistent_doc",
            ldu_store={}
        )
        
        assert isinstance(result, ProvenanceChain)
        assert "Error" in result.answer or "not found" in result.answer.lower()
        assert result.answer_confidence == 0.0
        assert result.verification_status == "unverifiable"
    
    def test_provenance_chain_structure(self, agent):
        """Test that ProvenanceChain has required fields."""
        result = agent.answer(
            query="Test",
            doc_id="test_doc",
            ldu_store={}
        )
        
        # Verify required fields exist
        assert hasattr(result, 'query')
        assert hasattr(result, 'answer')
        assert hasattr(result, 'answer_confidence')
        assert hasattr(result, 'citations')
        assert hasattr(result, 'retrieval_method')
        assert hasattr(result, 'verification_status')


class TestQueryState:
    """Test QueryState model."""
    
    def test_state_creation(self):
        """Test QueryState can be created."""
        state = QueryState(
            query="What is the profit?",
            doc_id="test_001"
        )
        
        assert state.query == "What is the profit?"
        assert state.doc_id == "test_001"
        assert state.retrieved_ldus == []
        assert state.confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
