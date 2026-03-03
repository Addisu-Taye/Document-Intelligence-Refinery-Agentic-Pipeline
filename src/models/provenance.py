# src/models/provenance.py
"""
Provenance Chain Schema - Audit Trail for Extracted Facts

Every answer from the Query Agent must include a ProvenanceChain that enables
verification against the source document. This is critical for enterprise trust.
"""

from pydantic import BaseModel, Field
from typing import Literal,  Optional
from datetime import datetime
from .ldu import BoundingBox


class ProvenanceCitation(BaseModel):
    """
    A single citation pointing to a source location in a document.
    
    This is the atomic unit of provenance - one specific location that
    supports part of an answer.
    """
    
    # === Document Identity ===
    document_name: str = Field(..., description="Original filename")
    doc_id: str = Field(..., description="Internal document identifier")
    
    # === Spatial Location ===
    page_number: int = Field(..., gt=0, description="Page number (1-indexed)")
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Exact coordinates of the cited content"
    )
    
    # === Content Verification ===
    content_hash: str = Field(..., description="SHA-256 hash of cited content")
    cited_text: str = Field(..., description="The exact text that supports the answer")
    
    # === Extraction Metadata ===
    extraction_strategy: Literal["fast_text", "layout_aware", "vision_augmented"]
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    
    # === Chunk Reference ===
    ldu_id: Optional[str] = Field(default=None, description="LDU that contains this content")
    section_path: list[str] = Field(
        default_factory=list,
        description="Hierarchy path to this content"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_name": "CBE ANNUAL REPORT 2023-24.pdf",
                "doc_id": "cbe_report_2024_abc123",
                "page_number": 42,
                "bounding_box": {"x0": 72.0, "y0": 450.2, "x1": 300.5, "y1": 480.1},
                "content_hash": "sha256:abc123...",
                "cited_text": "Net Profit: ETB 14.2 billion for FY 2023-24",
                "extraction_strategy": "layout_aware",
                "extraction_confidence": 0.94,
                "ldu_id": "ldu_042_001",
                "section_path": ["Financial Statements", "Financial Highlights"]
            }
        }


class ProvenanceChain(BaseModel):
    """
    Complete provenance trail for an answer.
    
    Contains all citations that support a query answer, enabling full
    audit and verification.
    """
    
    # === Query Context ===
    query: str = Field(..., description="The user's question")
    query_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # === Answer ===
    answer: str = Field(..., description="The generated answer")
    answer_confidence: float = Field(..., ge=0.0, le=1.0)
    
    # === Citations ===
    citations: list[ProvenanceCitation] = Field(
        ...,
        min_length=1,
        description="All source citations supporting this answer"
    )
    
    # === Retrieval Method ===
    retrieval_method: Literal["pageindex_navigation", "semantic_search", "structured_query", "hybrid"]
    retrieval_metadata: dict = Field(
        default_factory=dict,
        description="""
        Method-specific metadata:
        - pageindex_navigation: {sections_traversed: [...], nodes_visited: N}
        - semantic_search: {top_k: N, similarity_threshold: 0.7}
        - structured_query: {sql_query: "...", tables_queried: [...]}
        """
    )
    
    # === Verification Status ===
    verification_status: Literal["verified", "partial", "unverifiable"] = Field(
        default="verified",
        description="Whether all claims in answer are backed by citations"
    )
    unverifiable_claims: list[str] = Field(
        default_factory=list,
        description="Claims that could not be verified against source"
    )
    
    # === Cost Tracking ===
    tokens_used: int = Field(default=0, ge=0)
    cost_estimate_usd: float = Field(default=0.0, ge=0.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What was CBE's net profit in FY 2023-24?",
                "query_timestamp": "2024-01-15T14:30:00Z",
                "answer": "CBE's net profit was ETB 14.2 billion in FY 2023-24, representing a 10.9% increase from the previous year.",
                "answer_confidence": 0.96,
                "citations": [
                    {
                        "document_name": "CBE ANNUAL REPORT 2023-24.pdf",
                        "page_number": 42,
                        "bounding_box": {"x0": 72.0, "y0": 450.2, "x1": 300.5, "y1": 480.1},
                        "content_hash": "sha256:abc123...",
                        "cited_text": "Net Profit: ETB 14.2 billion",
                        "extraction_strategy": "layout_aware",
                        "extraction_confidence": 0.94
                    }
                ],
                "retrieval_method": "pageindex_navigation",
                "retrieval_metadata": {"sections_traversed": ["Financial Statements"], "nodes_visited": 3},
                "verification_status": "verified",
                "unverifiable_claims": [],
                "tokens_used": 245,
                "cost_estimate_usd": 0.0005
            }
        }


class AuditRecord(BaseModel):
    """
    Complete audit record for compliance and debugging.
    
    Combines the query, answer, provenance, and system state for full traceability.
    """
    
    audit_id: str = Field(..., description="Unique audit record identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # === User Context ===
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    
    # === Query & Answer ===
    query: str
    answer: str
    provenance_chain: ProvenanceChain
    
    # === System State ===
    pipeline_version: str = Field(default="1.0.0")
    models_used: list[str] = Field(default_factory=list)
    total_processing_time_seconds: float
    
    # === Compliance ===
    data_retention_days: int = Field(default=90)
    pii_detected: bool = Field(default=False)
    pii_redacted: bool = Field(default=False)
    
    class Config:
        json_schema_extra = {
            "example": {
                "audit_id": "audit_20240115_143000_abc123",
                "timestamp": "2024-01-15T14:30:00Z",
                "query": "What was CBE's net profit?",
                "answer": "ETB 14.2 billion",
                "provenance_chain": {...},
                "pipeline_version": "1.0.0",
                "models_used": ["gpt-4o-mini", "text-embedding-3-small"],
                "total_processing_time_seconds": 2.3,
                "data_retention_days": 90,
                "pii_detected": False,
                "pii_redacted": False
            }
        }
