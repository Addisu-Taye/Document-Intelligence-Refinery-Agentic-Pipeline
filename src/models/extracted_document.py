# src/models/extracted_document.py
"""
Extracted Document Schema - Normalized Output from Extraction Strategies

This is the unified schema that all three extraction strategies (Fast Text,
Layout-Aware, Vision-Augmented) must output. It enables the downstream
Chunking Engine to work regardless of which strategy was used.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal  # ← Literal added here
from datetime import datetime
from .ldu import BoundingBox


class ExtractedBlock(BaseModel):
    """
    A raw extracted content block before chunking.
    Used internally by extraction strategies.
    """
    block_id: str
    content: str
    block_type: Literal["text", "table", "figure", "header", "footer", "caption"]
    page: int
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0


class ExtractedDocument(BaseModel):
    """
    Normalized extraction output from any strategy.
    
    This schema wraps the raw extraction results and provides a unified
    interface for the Chunking Engine. All extraction strategies must
    produce output compatible with this schema.
    """
    
    # === Identity ===
    doc_id: str = Field(..., description="Matches DocumentProfile.doc_id")
    filename: str
    extraction_id: str = Field(..., description="Unique extraction run identifier")
    
    # === Extraction Metadata ===
    extraction_strategy: Literal["fast_text", "layout_aware", "vision_augmented"]
    extraction_started_at: datetime
    extraction_completed_at: datetime
    processing_time_seconds: float
    
    # === Quality Metrics ===
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    pages_processed: int
    pages_with_content: int
    pages_failed: list[int] = Field(default_factory=list)
    
    # === Raw Extracted Content (Pre-Chunking) ===
    blocks: list[ExtractedBlock] = Field(
        default_factory=list,
        description="Raw extracted blocks before semantic chunking"
    )
    
    # === Structured Tables ===
    tables: list[dict] = Field(
        default_factory=list,
        description="""
        Extracted tables as structured objects:
        {
            "table_id": "tbl_001",
            "page": 42,
            "bbox": {...},
            "headers": ["Column1", "Column2"],
            "rows": [["val1", "val2"], ...],
            "caption": "Table 3: Financial Summary"
        }
        """
    )
    
    # === Figures with Captions ===
    figures: list[dict] = Field(
        default_factory=list,
        description="""
        Extracted figures:
        {
            "figure_id": "fig_001",
            "page": 15,
            "bbox": {...},
            "caption": "Figure 2: Revenue Growth",
            "type": "chart"
        }
        """
    )
    
    # === Reading Order ===
    reading_order: list[str] = Field(
        default_factory=list,
        description="Ordered list of block_ids representing document reading flow"
    )
    
    # === Cost Tracking ===
    cost_estimate_usd: float = Field(default=0.0, ge=0.0)
    tokens_used: int = Field(default=0, ge=0)
    
    # === Escalation History ===
    escalation_count: int = Field(
        default=0,
        description="Number of strategy escalations during extraction"
    )
    escalation_log: list[dict] = Field(
        default_factory=list,
        description="""
        Log of escalation decisions:
        {
            "from_strategy": "fast_text",
            "to_strategy": "layout_aware",
            "reason": "confidence 0.62 < threshold 0.75",
            "pages": [1, 2, 3]
        }
        """
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "cbe_report_2024_abc123",
                "filename": "CBE ANNUAL REPORT 2023-24.pdf",
                "extraction_id": "ext_xyz789",
                "extraction_strategy": "layout_aware",
                "extraction_started_at": "2024-01-15T10:30:00Z",
                "extraction_completed_at": "2024-01-15T10:32:45Z",
                "processing_time_seconds": 165.0,
                "overall_confidence": 0.92,
                "pages_processed": 161,
                "pages_with_content": 158,
                "pages_failed": [],
                "tables": [{"table_id": "tbl_001", "page": 42, "headers": [...], "rows": [...]}],
                "figures": [{"figure_id": "fig_001", "page": 15, "caption": "Revenue Growth"}],
                "cost_estimate_usd": 0.0,
                "tokens_used": 0,
                "escalation_count": 1,
                "escalation_log": [{"from_strategy": "fast_text", "to_strategy": "layout_aware", "reason": "low confidence"}]
            }
        }