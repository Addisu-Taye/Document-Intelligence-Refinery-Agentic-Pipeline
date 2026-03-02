# src/models/__init__.py
"""
Core Data Models for Document Intelligence Refinery

All Pydantic schemas used across the pipeline stages.
"""

from .document_profile import DocumentProfile
from .ldu import LDU, BoundingBox
from .extracted_document import ExtractedDocument, ExtractedBlock
from .page_index import PageIndex, PageIndexNode
from .provenance import ProvenanceChain, ProvenanceCitation, AuditRecord

__all__ = [
    # Triage Agent
    "DocumentProfile",
    
    # Chunking Engine
    "LDU",
    "BoundingBox",
    
    # Extraction Layer
    "ExtractedDocument",
    "ExtractedBlock",
    
    # PageIndex Builder
    "PageIndex",
    "PageIndexNode",
    
    # Provenance Layer
    "ProvenanceChain",
    "ProvenanceCitation",
    "AuditRecord",
]

# Version tracking for audit
__version__ = "1.0.0"