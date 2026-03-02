# src/models/ldu.py
"""
Logical Document Unit (LDU) Schema - Output of Semantic Chunking Engine

LDUs are semantically coherent, self-contained units that preserve structural context.
This is the fundamental unit for RAG retrieval and provenance tracking.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
import hashlib


class BoundingBox(BaseModel):
    """
    Spatial coordinates for a content element on a page.
    Coordinate system: points (1/72 inch), origin at bottom-left.
    """
    x0: float = Field(..., description="Left edge coordinate")
    y0: float = Field(..., description="Bottom edge coordinate")
    x1: float = Field(..., description="Right edge coordinate")
    y1: float = Field(..., description="Top edge coordinate")
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


class LDU(BaseModel):
    """
    Logical Document Unit - A semantically coherent chunk of document content.
    
    Chunking Constitution (5 Rules):
    1. Table cells never split from header rows
    2. Figure captions stored as parent metadata
    3. Numbered lists kept atomic (unless > max_tokens)
    4. Section headers propagated to child chunks
    5. Cross-references resolved as chunk relationships
    """
    
    # === Identity ===
    ldu_id: str = Field(..., description="Unique LDU identifier (hash of content + bbox + page)")
    content: str = Field(..., description="The actual text/content of this LDU")
    
    # === Classification ===
    chunk_type: Literal["text_block", "table", "figure", "list", "section_header", "equation", "mixed"] = Field(
        ...,
        description="Type of content in this LDU"
    )
    
    # === Spatial Provenance (Critical for Audit) ===
    page_refs: list[int] = Field(..., description="Page numbers this LDU spans (1-indexed)")
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box on the primary page (first page in page_refs)"
    )
    bounding_boxes: list[BoundingBox] = Field(
        default_factory=list,
        description="All bounding boxes if LDU spans multiple pages"
    )
    
    # === Hierarchical Context ===
    parent_section: Optional[str] = Field(
        default=None,
        description="Section title this LDU belongs to (for navigation)"
    )
    section_path: list[str] = Field(
        default_factory=list,
        description="Full hierarchy path: ['Chapter 1', 'Section 2.1', 'Subsection 2.1.3']"
    )
    
    # === Content Metadata ===
    token_count: int = Field(..., ge=0, description="Token count for this LDU (for RAG limits)")
    word_count: int = Field(default=0, ge=0)
    language: str = Field(default="en")
    
    # === Table-Specific Fields (if chunk_type == "table") ===
    table_headers: Optional[list[str]] = Field(
        default=None,
        description="Column headers for table chunks"
    )
    table_row_count: Optional[int] = Field(default=None, ge=0)
    table_data: Optional[list[dict]] = Field(
        default=None,
        description="Structured table data as list of row dicts"
    )
    
    # === Figure-Specific Fields (if chunk_type == "figure") ===
    figure_caption: Optional[str] = Field(default=None)
    figure_type: Optional[Literal["chart", "graph", "diagram", "photo", "screenshot"]] = Field(default=None)
    
    # === Relationships ===
    cross_references: list[str] = Field(
        default_factory=list,
        description="LDU IDs this chunk references (e.g., 'see Table 3' resolved to LDU ID)"
    )
    referenced_by: list[str] = Field(
        default_factory=list,
        description="LDU IDs that reference this chunk"
    )
    
    # === Integrity ===
    content_hash: str = Field(..., description="SHA-256 hash of content for provenance verification")
    extraction_strategy: Literal["fast_text", "layout_aware", "vision_augmented"] = Field(
        ...,
        description="Which extraction strategy produced this LDU"
    )
    extraction_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from extraction strategy"
    )
    
    @classmethod
    def compute_content_hash(cls, content: str, page_refs: list[int], bbox: Optional[BoundingBox]) -> str:
        """Generate deterministic hash for provenance verification."""
        data = f"{content}|{page_refs}|{bbox.to_tuple() if bbox else 'None'}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    class Config:
        json_schema_extra = {
            "example": {
                "ldu_id": "ldu_abc123...",
                "content": "Net Profit: ETB 14.2 billion for FY 2023-24",
                "chunk_type": "table",
                "page_refs": [42],
                "bounding_box": {"x0": 72.0, "y0": 450.2, "x1": 300.5, "y1": 480.1},
                "parent_section": "Financial Highlights",
                "section_path": ["Annual Report", "Financial Statements", "Financial Highlights"],
                "token_count": 12,
                "table_headers": ["Metric", "FY 2023-24", "FY 2022-23"],
                "table_data": [{"Metric": "Net Profit", "FY 2023-24": "14.2B", "FY 2022-23": "12.8B"}],
                "content_hash": "sha256:abc123...",
                "extraction_strategy": "layout_aware",
                "extraction_confidence": 0.94
            }
        }