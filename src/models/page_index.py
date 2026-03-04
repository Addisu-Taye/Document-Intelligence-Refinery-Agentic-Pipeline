# src/models/page_index.py
"""
PageIndex Schema - Hierarchical Navigation Tree

Inspired by VectifyAI's PageIndex, this provides a "smart table of contents"
that enables LLMs to navigate documents without reading every page.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal
from datetime import datetime


class PageIndexNode(BaseModel):
    """
    A node in the PageIndex hierarchical tree.
    
    Each node represents a section/subsection with metadata that enables
    efficient navigation and retrieval.
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "node_id": "section_2_1",
            "title": "Financial Highlights",
            "level": 2,
            "parent_node_id": "chapter_2",
            "child_node_ids": ["section_2_1_1", "section_2_1_2"],
            "page_start": 40,
            "page_end": 55,
            "summary": "Contains key financial metrics including net profit, total assets, and capital adequacy ratios for FY 2023-24.",
            "key_entities": ["ETB 14.2 billion", "Commercial Bank of Ethiopia", "NBE"],
            "data_types_present": ["tables", "figures", "text"],
            "table_count": 8,
            "figure_count": 3,
            "keywords": ["profit", "revenue", "assets", "financial"],
            "ldu_ids": ["ldu_001", "ldu_002", "ldu_003"]
        }
    })
    
    # === Identity ===
    node_id: str = Field(..., description="Unique node identifier")
    title: str = Field(..., description="Section title")
    
    # === Hierarchy ===
    level: int = Field(..., ge=0, description="Depth in hierarchy (0=root)")
    parent_node_id: Optional[str] = Field(default=None)
    child_node_ids: list[str] = Field(default_factory=list)
    
    # === Page Coverage ===
    page_start: int = Field(..., gt=0, description="First page of this section (1-indexed)")
    page_end: int = Field(..., gt=0, description="Last page of this section")
    
    # === Content Summary (LLM-Generated) ===
    summary: str = Field(default="", description="2-3 sentence summary of section content")
    summary_model: str = Field(default="gpt-4o-mini", description="Model used to generate summary")
    summary_tokens: int = Field(default=0, ge=0, description="Token count of summary")
    
    # === Content Inventory ===
    key_entities: list[str] = Field(default_factory=list, description="Named entities found in this section")
    data_types_present: list[Literal["tables", "figures", "equations", "lists", "text"]] = Field(default_factory=list)
    table_count: int = Field(default=0, ge=0)
    figure_count: int = Field(default=0, ge=0)
    
    # === Navigation Hints ===
    keywords: list[str] = Field(default_factory=list, description="Search keywords for this section")
    relevance_scores: dict = Field(default_factory=dict, description="Pre-computed relevance scores for common query types")
    
    # === LDU References ===
    ldu_ids: list[str] = Field(default_factory=list, description="LDU IDs that belong to this section")
    
    # === Metadata ===
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)


class PageIndex(BaseModel):
    """
    Complete PageIndex tree for a document.
    
    This is the root container that holds the entire navigation hierarchy.
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "doc_id": "cbe_report_2024_abc123",
            "filename": "CBE ANNUAL REPORT 2023-24.pdf",
            "root_nodes": [{"node_id": "chapter_1", "title": "Executive Summary", "level": 0}],
            "all_nodes": {"chapter_1": {...}, "section_1_1": {...}},
            "total_nodes": 47,
            "max_depth": 4,
            "total_pages_covered": 161,
            "summary_model_used": "gpt-4o-mini",
            "generation_time_seconds": 45.2
        }
    })
    
    # === Identity ===
    doc_id: str = Field(..., description="Matches DocumentProfile.doc_id")
    filename: str
    pageindex_created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # === Tree Structure ===
    root_nodes: list[PageIndexNode] = Field(default_factory=list, description="Top-level sections (level 0)")
    all_nodes: dict[str, PageIndexNode] = Field(default_factory=dict, description="Flat lookup: node_id -> PageIndexNode")
    
    # === Statistics ===
    total_nodes: int = Field(..., ge=0)
    max_depth: int = Field(..., ge=0)
    total_pages_covered: int = Field(..., ge=0)
    
    # === Generation Metadata ===
    summary_model_used: str
    generation_time_seconds: float
    
    # === Query Helpers ===
    def get_node(self, node_id: str) -> Optional[PageIndexNode]:
        """Get a node by ID."""
        return self.all_nodes.get(node_id)
    
    def get_children(self, node_id: str) -> list[PageIndexNode]:
        """Get child nodes of a given node."""
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.all_nodes[cid] for cid in node.child_node_ids if cid in self.all_nodes]
    
    def get_ancestors(self, node_id: str) -> list[PageIndexNode]:
        """Get ancestor nodes (parent chain) of a given node."""
        ancestors = []
        current = self.get_node(node_id)
        while current and current.parent_node_id:
            parent = self.get_node(current.parent_node_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors
    
    def search_by_keyword(self, keyword: str) -> list[PageIndexNode]:
        """Find nodes that contain a keyword in title, summary, or keywords."""
        keyword_lower = keyword.lower()
        results = []
        for node in self.all_nodes.values():
            if (keyword_lower in node.title.lower() or 
                keyword_lower in node.summary.lower() or
                any(keyword_lower in k.lower() for k in node.keywords)):
                results.append(node)
        return results
    
    def get_ldus_for_section(self, node_id: str) -> list[str]:
        """Get all LDU IDs for a section and its children."""
        node = self.get_node(node_id)
        if not node:
            return []
        
        ldu_ids = list(node.ldu_ids)
        for child_id in node.child_node_ids:
            ldu_ids.extend(self.get_ldus_for_section(child_id))
        return ldu_ids
