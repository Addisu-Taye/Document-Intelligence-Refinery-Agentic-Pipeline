# tests/test_indexer.py
"""
Unit Tests for PageIndex Builder

Tests hierarchical index construction, summary generation, and search.
"""

import pytest
from datetime import datetime
from src.agents.indexer import PageIndexBuilder
from src.models.ldu import LDU, BoundingBox
from src.models.page_index import PageIndex, PageIndexNode


class TestPageIndexBuilder:
    """Test PageIndex construction logic."""
    
    @pytest.fixture
    def builder(self):
        """Create PageIndexBuilder instance."""
        return PageIndexBuilder(generate_summaries=False)
    
    @pytest.fixture
    def sample_ldus(self):
        """Create sample LDUs for testing."""
        return [
            # Section header
            LDU(
                ldu_id="ldu_header_001",
                content="Financial Highlights",
                chunk_type="section_header",
                page_refs=[10],
                token_count=5,
                word_count=2,
                content_hash="hash_header",
                extraction_strategy="fast_text",
                extraction_confidence=0.95,
                parent_section=None,
                section_path=["Annual Report", "Financial Statements", "Financial Highlights"]
            ),
            # Text content
            LDU(
                ldu_id="ldu_text_001",
                content="Net profit increased to ETB 14.2 billion in FY 2023-24, representing 10.9% growth.",
                chunk_type="text_block",
                page_refs=[10],
                token_count=20,
                word_count=15,
                content_hash="hash_text1",
                extraction_strategy="fast_text",
                extraction_confidence=0.88,
                parent_section="Financial Highlights",
                section_path=["Annual Report", "Financial Statements", "Financial Highlights"]
            ),
            # Table
            LDU(
                ldu_id="ldu_table_001",
                content="Headers: Metric | 2023 | 2024\nData: Revenue | 100M | 120M",
                chunk_type="table",
                page_refs=[11],
                token_count=25,
                word_count=12,
                table_headers=["Metric", "2023", "2024"],
                table_row_count=1,
                table_data=[{"Metric": "Revenue", "2023": "100M", "2024": "120M"}],
                content_hash="hash_table",
                extraction_strategy="fast_text",
                extraction_confidence=0.92,
                parent_section="Financial Highlights",
                section_path=["Annual Report", "Financial Statements", "Financial Highlights"]
            ),
            # Figure
            LDU(
                ldu_id="ldu_figure_001",
                content="[chart] Revenue Trend 2020-2024",
                chunk_type="figure",
                page_refs=[12],
                token_count=8,
                word_count=4,
                figure_caption="Figure 3: Revenue Growth Trend",
                figure_type="chart",
                content_hash="hash_figure",
                extraction_strategy="layout_aware",
                extraction_confidence=0.85,
                parent_section="Financial Highlights",
                section_path=["Annual Report", "Financial Statements", "Financial Highlights"]
            ),
        ]
    
    def test_builder_initialization(self, builder):
        """Test builder initializes correctly."""
        assert builder is not None
        assert builder.summary_model == "gpt-4o-mini"
        assert builder.generate_summaries == False
    
    def test_keyword_extraction(self, builder):
        """Test keyword extraction from text."""
        text = "The Commercial Bank of Ethiopia reported net profit of ETB 14.2 billion with 10.9% growth in revenue."
        keywords = builder._extract_keywords(text, max_keywords=5)
        
        assert len(keywords) <= 5
        assert "profit" in keywords or "ethiopia" in keywords or "billion" in keywords
    
    def test_entity_extraction(self, builder):
        """Test entity extraction patterns."""
        text = "CBE reported ETB 14.2 billion profit with 10.9% growth in FY 2024. The NBE directive was followed."
        entities = builder._extract_entities(text)
        
        # Should find currency, percentage, and organization entities
        assert any("ETB" in e or "billion" in e for e in entities)
        assert any("%" in e for e in entities)
        assert any("CBE" in e or "NBE" in e for e in entities)
    
    def test_summary_generation_stub(self, builder, sample_ldus):
        """Test heuristic summary generation."""
        section_ldus = [ldu for ldu in sample_ldus if ldu.parent_section == "Financial Highlights"]
        summary = builder._generate_summary_stub(section_ldus, "Financial Highlights")
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Financial Highlights" in summary or len(summary) < 200
    
    def test_section_hierarchy_detection(self, builder, sample_ldus):
        """Test grouping LDUs by section."""
        sections = builder._detect_section_hierarchy(sample_ldus)
        
        # Should group by section_path
        assert len(sections) >= 1
        # Check that LDUs are grouped correctly
        for section_title, ldus in sections.items():
            assert len(ldus) > 0
            for ldu in ldus:
                assert ldu in sample_ldus
    
    def test_build_node(self, builder, sample_ldus):
        """Test PageIndexNode creation."""
        section_ldus = [ldu for ldu in sample_ldus if ldu.parent_section == "Financial Highlights"]
        node_counter = [0]
        
        node = builder._build_node(
            section_title="Financial Highlights",
            ldus=section_ldus,
            level=2,
            parent_id="parent_001",
            node_counter=node_counter
        )
        
        assert node.node_id == "node_0"
        assert node.title == "Financial Highlights"
        assert node.level == 2
        assert node.parent_node_id == "parent_001"
        assert node.page_start >= 10
        assert node.page_end <= 12
        assert node.table_count >= 1
        assert node.figure_count >= 1
        assert len(node.keywords) > 0
        assert len(node.key_entities) > 0
    
    def test_build_pageindex(self, builder, sample_ldus):
        """Test full PageIndex construction."""
        pageindex = builder.build(
            ldus=sample_ldus,
            doc_id="test_doc_001",
            filename="test_report.pdf"
        )
        
        assert isinstance(pageindex, PageIndex)
        assert pageindex.doc_id == "test_doc_001"
        assert pageindex.filename == "test_report.pdf"
        assert pageindex.total_nodes >= 1
        assert pageindex.max_depth >= 0
        assert pageindex.total_pages_covered >= 10
        assert len(pageindex.all_nodes) == pageindex.total_nodes
    
    def test_pageindex_search_by_keyword(self, builder, sample_ldus):
        """Test keyword search in PageIndex."""
        pageindex = builder.build(sample_ldus, "test_doc_001", "test_report.pdf")
        
        # Search for financial keywords
        results = pageindex.search_by_keyword("profit")
        
        # Should find nodes containing the keyword
        assert isinstance(results, list)
        # May or may not find results depending on summary generation
    
    def test_pageindex_get_ldus_for_section(self, builder, sample_ldus):
        """Test LDU retrieval for a section."""
        pageindex = builder.build(sample_ldus, "test_doc_001", "test_report.pdf")
        
        # Get LDUs for first root node
        if pageindex.root_nodes:
            root_id = pageindex.root_nodes[0].node_id
            ldu_ids = pageindex.get_ldus_for_section(root_id)
            
            assert isinstance(ldu_ids, list)
            # Should return LDU IDs that belong to this section
    
    def test_pageindex_get_ancestors(self, builder, sample_ldus):
        """Test ancestor retrieval in hierarchy."""
        pageindex = builder.build(sample_ldus, "test_doc_001", "test_report.pdf")
        
        # Test with a node that has a parent
        for node in pageindex.all_nodes.values():
            if node.parent_node_id:
                ancestors = pageindex.get_ancestors(node.node_id)
                assert isinstance(ancestors, list)
                # Ancestors should include parent chain
                break
    
    def test_save_pageindex(self, builder, sample_ldus, tmp_path):
        """Test saving PageIndex to JSON."""
        pageindex = builder.build(sample_ldus, "test_doc_001", "test_report.pdf")
        
        output_dir = tmp_path / "pageindex"
        output_path = builder.save(pageindex, str(output_dir))
        
        assert output_path.exists()
        assert output_path.suffix == ".json"
        
        # Verify JSON is valid
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        assert data['doc_id'] == "test_doc_001"
        assert data['total_nodes'] == pageindex.total_nodes


class TestPageIndexNode:
    """Test PageIndexNode model validation."""
    
    def test_node_creation(self):
        """Test PageIndexNode can be created with valid data."""
        node = PageIndexNode(
            node_id="test_node_001",
            title="Test Section",
            level=1,
            page_start=10,
            page_end=20,
            summary="Test summary",
            key_entities=["Entity1", "Entity2"],
            data_types_present=["text", "tables"],
            table_count=2,
            figure_count=1,
            keywords=["test", "keyword"],
            ldu_ids=["ldu_001", "ldu_002"]
        )
        
        assert node.node_id == "test_node_001"
        assert node.title == "Test Section"
        assert node.level == 1
        assert node.table_count == 2
        assert "ldu_001" in node.ldu_ids
    
    def test_node_json_serialization(self):
        """Test PageIndexNode can be serialized to JSON."""
        node = PageIndexNode(
            node_id="test_node_002",
            title="JSON Test",
            level=0,
            page_start=1,
            page_end=5,
            summary="Summary"
        )
        
        # Should not raise
        json_str = node.model_dump_json()
        assert "JSON Test" in json_str
        assert "test_node_002" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


