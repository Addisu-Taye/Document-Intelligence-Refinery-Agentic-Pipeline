# src/models/provenance.py
"""
Provenance Chain Schema - Audit Trail for Extracted Facts
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal
from datetime import datetime


class ProvenanceCitation(BaseModel):
    """A single citation pointing to a source location in a document."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "document_name": "CBE ANNUAL REPORT 2023-24.pdf",
            "doc_id": "cbe_report_2024_abc123",
            "page_number": 42,
            "bounding_box": {"x0": 72.0, "y0": 450.2, "x1": 300.5, "y1": 480.1},
            "content_hash": "sha256:abc123...",
            "cited_text": "Net Profit: ETB 14.2 billion",
            "extraction_strategy": "layout_aware",
            "extraction_confidence": 0.94
        }
    })
    
    document_name: str = Field(..., description="Original filename")
    doc_id: str = Field(..., description="Internal document identifier")
    page_number: int = Field(..., gt=0, description="Page number (1-indexed)")
    bounding_box: Optional[dict] = Field(default=None)
    content_hash: str = Field(..., description="SHA-256 hash of cited content")
    cited_text: str = Field(..., description="The exact text that supports the answer")
    extraction_strategy: Literal["fast_text", "layout_aware", "vision_augmented"]
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    ldu_id: Optional[str] = Field(default=None)
    section_path: list[str] = Field(default_factory=list)


class ProvenanceChain(BaseModel):
    """Complete provenance trail for an answer."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "What was CBE's net profit?",
            "answer": "ETB 14.2 billion",
            "answer_confidence": 0.96,
            "citations": [],
            "retrieval_method": "pageindex_navigation",
            "verification_status": "verified"
        }
    })
    
    query: str = Field(..., description="The user's question")
    query_timestamp: datetime = Field(default_factory=datetime.utcnow)
    answer: str = Field(..., description="The generated answer")
    answer_confidence: float = Field(..., ge=0.0, le=1.0)
    
    # === FIXED: Allow empty citations for error cases ===
    citations: list[ProvenanceCitation] = Field(
        default_factory=list,
        min_length=0,
        description="Source citations (empty if no content found)"
    )
    
    retrieval_method: Literal["pageindex_navigation", "semantic_search", "structured_query", "hybrid"]
    retrieval_metadata: dict = Field(default_factory=dict)
    verification_status: Literal["verified", "partial", "unverifiable"] = Field(default="verified")
    unverifiable_claims: list[str] = Field(default_factory=list)
    tokens_used: int = Field(default=0, ge=0)
    cost_estimate_usd: float = Field(default=0.0, ge=0.0)


class AuditRecord(BaseModel):
    """Complete audit record for compliance and debugging."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "audit_id": "audit_001",
            "query": "Test query",
            "answer": "Test answer",
            "provenance_chain": {},
            "pipeline_version": "1.0.0"
        }
    })
    
    audit_id: str = Field(..., description="Unique audit record identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    query: str
    answer: str
    provenance_chain: ProvenanceChain
    pipeline_version: str = Field(default="1.0.0")
    models_used: list[str] = Field(default_factory=list)
    total_processing_time_seconds: float
    data_retention_days: int = Field(default=90)
    pii_detected: bool = Field(default=False)
    pii_redacted: bool = Field(default=False)
