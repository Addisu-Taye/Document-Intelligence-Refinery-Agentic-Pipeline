# tests/test_chunker.py
"""
Unit Tests for Semantic Chunking Engine
"""

import pytest
from datetime import datetime
from src.agents.chunker import SemanticChunker, ChunkingConfig, ChunkingResult
from src.agents.validator import ChunkValidator
from src.models.ldu import LDU, BoundingBox
from src.models.extracted_document import ExtractedDocument, ExtractedBlock


class TestChunkValidator:
    @pytest.fixture
    def validator(self):
        return ChunkValidator()
    
    def test_table_with_headers_valid(self, validator):
        ldu = LDU(
            ldu_id="test_tbl_001",
            content="Headers: Col1 | Col2\nData: A | B",
            chunk_type="table",
            page_refs=[1],
            bounding_box=BoundingBox(x0=0, y0=0, x1=100, y1=50),
            parent_section="Financial Data",
            section_path=["Report", "Financial Data"],
            token_count=50,
            word_count=12,
            table_headers=["Col1", "Col2"],
            table_row_count=1,
            table_data=[{"Col1": "A", "Col2": "B"}],
            content_hash="abc123",
            extraction_strategy="fast_text",
            extraction_confidence=0.85
        )
        is_valid, errors = validator.validate(ldu)
        assert is_valid == True
        assert len(errors) == 0
    
    def test_table_without_headers_invalid(self, validator):
        ldu = LDU(
            ldu_id="test_tbl_002",
            content="Data: A | B",
            chunk_type="table",
            page_refs=[1],
            token_count=20,
            word_count=5,
            table_headers=None,
            table_row_count=1,
            content_hash="def456",
            extraction_strategy="fast_text",
            extraction_confidence=0.80
        )
        is_valid, errors = validator.validate(ldu)
        assert is_valid == False
        assert len(errors) > 0
        assert "headers" in errors[0].lower()
    
    def test_figure_with_caption_valid(self, validator):
        ldu = LDU(
            ldu_id="test_fig_001",
            content="[chart] Revenue Growth 2020-2024",
            chunk_type="figure",
            page_refs=[5],
            token_count=15,
            word_count=4,
            figure_caption="Figure 3: Revenue Growth 2020-2024",
            figure_type="chart",
            content_hash="ghi789",
            extraction_strategy="layout_aware",
            extraction_confidence=0.88
        )
        is_valid, errors = validator.validate(ldu)
        assert is_valid == True
    
    def test_figure_without_caption_invalid(self, validator):
        ldu = LDU(
            ldu_id="test_fig_002",
            content="[chart]",
            chunk_type="figure",
            page_refs=[5],
            token_count=5,
            word_count=1,
            figure_caption=None,
            content_hash="jkl012",
            extraction_strategy="layout_aware",
            extraction_confidence=0.75
        )
        is_valid, errors = validator.validate(ldu)
        assert is_valid == False
        assert len(errors) > 0
    
    def test_list_within_token_limit_valid(self, validator):
        ldu = LDU(
            ldu_id="test_list_001",
            content="1. Item one\n2. Item two",
            chunk_type="list",
            page_refs=[2],
            token_count=100,
            word_count=15,
            content_hash="mno345",
            extraction_strategy="fast_text",
            extraction_confidence=0.82
        )
        is_valid, errors = validator.validate(ldu)
        assert is_valid == True
    
    def test_list_exceeds_token_limit_warning(self, validator):
        ldu = LDU(
            ldu_id="test_list_002",
            content="1. " + "item " * 200,
            chunk_type="list",
            page_refs=[2],
            token_count=600,
            word_count=200,
            content_hash="pqr678",
            extraction_strategy="fast_text",
            extraction_confidence=0.78
        )
        is_valid, errors = validator.validate(ldu)
        assert is_valid == True
    
    def test_section_propagation_valid(self, validator):
        ldu = LDU(
            ldu_id="test_txt_001",
            content="Financial content",
            chunk_type="text_block",
            page_refs=[10],
            parent_section="Financial Highlights",
            section_path=["Report", "Financial Highlights"],
            token_count=25,
            word_count=8,
            content_hash="stu901",
            extraction_strategy="fast_text",
            extraction_confidence=0.85
        )
        is_valid, errors = validator.validate(ldu)
        assert is_valid == True
    
    def test_validate_batch_summary(self, validator):
        ldus = [
            LDU(
                ldu_id=f"test_{i}",
                content="Test content",
                chunk_type="text_block",
                page_refs=[1],
                token_count=20,
                word_count=5,
                content_hash=f"hash_{i}",
                extraction_strategy="fast_text",
                extraction_confidence=0.85
            )
            for i in range(10)
        ]
        summary = validator.validate_batch(ldus)
        assert summary["total_ldus"] == 10
        assert summary["passed"] == 10
        assert summary["pass_rate"] == 1.0


class TestSemanticChunker:
    @pytest.fixture
    def chunker(self):
        config = ChunkingConfig()
        return SemanticChunker(config=config)
    
    @pytest.fixture
    def sample_extracted_doc(self):
        blocks = [
            ExtractedBlock(
                block_id="blk_001",
                content="Executive Summary",
                block_type="header",
                page=1,
                bbox=BoundingBox(x0=72, y0=720, x1=300, y1=750),
                confidence=0.95
            ),
            ExtractedBlock(
                block_id="blk_002",
                content="This report provides an overview.",
                block_type="text",
                page=1,
                bbox=BoundingBox(x0=72, y0=650, x1=500, y1=710),
                confidence=0.88
            )
        ]
        tables = [{
            "table_id": "tbl_001",
            "page": 2,
            "bbox": {"x0": 72, "y0": 400, "x1": 500, "y1": 500},
            "headers": ["Metric", "2023", "2024"],
            "rows": [["Revenue", "100M", "120M"]],
            "caption": "Table 1: Financial Summary",
            "row_count": 1,
            "column_count": 3
        }]
        figures = [{
            "figure_id": "fig_001",
            "page": 3,
            "bbox": {"x0": 100, "y0": 300, "x1": 400, "y1": 500},
            "caption": "Figure 1: Revenue Trend",
            "type": "chart"
        }]
        return ExtractedDocument(
            doc_id="test_doc_001",
            filename="test_report.pdf",
            extraction_id="ext_test_001",
            extraction_strategy="fast_text",
            extraction_started_at=datetime(2024, 1, 15, 10, 30, 0),
            extraction_completed_at=datetime(2024, 1, 15, 10, 32, 45),
            processing_time_seconds=1.5,
            overall_confidence=0.88,
            pages_processed=5,
            pages_with_content=5,
            pages_failed=[],
            blocks=blocks,
            tables=tables,
            figures=figures,
            reading_order=["blk_001", "blk_002"],
            cost_estimate_usd=0.0,
            tokens_used=0,
            escalation_count=0,
            escalation_log=[]
        )
    
    def test_chunker_initialization(self, chunker):
        assert chunker is not None
        assert chunker.config.max_tokens_per_ldu == 512
    
    def test_chunk_extracted_document(self, chunker, sample_extracted_doc):
        result = chunker.chunk(sample_extracted_doc)
        assert isinstance(result, ChunkingResult)
        assert result.total_ldus > 0
    
    def test_chunking_creates_table_ldu(self, chunker, sample_extracted_doc):
        result = chunker.chunk(sample_extracted_doc)
        table_ldus = [ldu for ldu in result.ldus if ldu.chunk_type == "table"]
        assert len(table_ldus) > 0
    
    def test_chunking_creates_figure_ldu(self, chunker, sample_extracted_doc):
        result = chunker.chunk(sample_extracted_doc)
        figure_ldus = [ldu for ldu in result.ldus if ldu.chunk_type == "figure"]
        assert len(figure_ldus) > 0
    
    def test_content_hash_generation(self, chunker, sample_extracted_doc):
        result = chunker.chunk(sample_extracted_doc)
        hashes = [ldu.content_hash for ldu in result.ldus]
        assert len(hashes) == len(set(hashes))
    
    def test_validation_integration(self, chunker, sample_extracted_doc):
        result = chunker.chunk(sample_extracted_doc)
        assert hasattr(result, 'validation_passed')


class TestChunkingConfig:
    def test_default_config(self):
        config = ChunkingConfig()
        assert config.max_tokens_per_ldu == 512
    
    def test_custom_config(self):
        config = ChunkingConfig(max_tokens_per_ldu=256)
        assert config.max_tokens_per_ldu == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
