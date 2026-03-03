# src/strategies/layout_aware.py
"""
Strategy B: Layout-Aware Extraction

Uses Docling or MinerU for documents with complex layouts (multi-column, tables, figures).
Higher quality than Strategy A, but slower and requires additional dependencies.

Triggers when:
- layout_complexity = multi_column | table_heavy | mixed
- Strategy A confidence < threshold
- origin_type = mixed
"""

import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field

from src.models.extracted_document import ExtractedDocument, ExtractedBlock, BoundingBox
from src.models.document_profile import DocumentProfile


class LayoutExtractor:
    """
    Strategy B: Layout-aware extraction using Docling/MinerU.
    
    This strategy handles complex layouts (multi-column, tables, figures) that
    break Strategy A. It extracts structure with bounding boxes and reading order.
    
    Note: This is a stub implementation. In production, you would integrate
    actual Docling or MinerU libraries here.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize with configuration from extraction_rules.yaml.
        
        Args:
            config: Configuration dict with confidence thresholds
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get('layout_aware', {}).get('min_confidence_to_proceed', 0.80)
        
        # Try to import Docling (optional dependency)
        self._docling_available = False
        try:
            # Uncomment when Docling is installed:
            # from docling.document_converter import DocumentConverter
            # self._docling_available = True
            pass
        except ImportError:
            pass
    
    def _extract_with_docling(self, pdf_path: Path) -> dict:
        """
        Extract using Docling (if available).
        
        Returns normalized extraction result.
        """
        # Stub implementation - would use actual Docling in production
        # from docling.document_converter import DocumentConverter
        # converter = DocumentConverter()
        # result = converter.convert(str(pdf_path))
        # return self._normalize_docling_output(result)
        
        return {
            'blocks': [],
            'tables': [],
            'figures': [],
            'reading_order': [],
            'confidence': 0.85  # Default confidence for layout-aware
        }
    
    def _extract_with_mineru(self, pdf_path: Path) -> dict:
        """
        Extract using MinerU (if available).
        
        Returns normalized extraction result.
        """
        # Stub implementation - would use actual MinerU in production
        return {
            'blocks': [],
            'tables': [],
            'figures': [],
            'reading_order': [],
            'confidence': 0.85
        }
    
    def _extract_with_fallback(self, pdf_path: Path) -> dict:
        """
        Fallback extraction using pdfplumber with layout heuristics.
        
        Used when Docling/MinerU are not available.
        """
        import pdfplumber
        import hashlib
        
        blocks = []
        tables = []
        figures = []
        reading_order = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_num = page.page_number
                
                # Extract text with better structure awareness
                text = page.extract_text() or ""
                if text.strip():
                    block = ExtractedBlock(
                        block_id=f"blk_layout_p{page_num}",
                        content=text,
                        block_type="text",
                        page=page_num,
                        bbox=None,
                        confidence=0.80
                    )
                    blocks.append(block)
                    reading_order.append(block.block_id)
                
                # Extract tables
                try:
                    pdf_tables = page.find_tables()
                    for i, table in enumerate(pdf_tables):
                        table_data = table.extract()
                        if table_data and len(table_data) >= 2:
                            headers = [str(h) if h else f"col_{j}" for j, h in enumerate(table_data[0])]
                            rows = [[str(cell) if cell else "" for cell in row] for row in table_data[1:]]
                            
                            table_obj = {
                                'table_id': f"tbl_layout_p{page_num}_t{i}",
                                'page': page_num,
                                'bbox': {'x0': table.bbox[0], 'y0': table.bbox[1], 'x1': table.bbox[2], 'y1': table.bbox[3]} if hasattr(table, 'bbox') and table.bbox else None,
                                'headers': headers,
                                'rows': rows,
                                'caption': None,
                                'row_count': len(rows),
                                'column_count': len(headers)
                            }
                            tables.append(table_obj)
                except Exception:
                    pass
        
        return {
            'blocks': blocks,
            'tables': tables,
            'figures': figures,
            'reading_order': reading_order,
            'confidence': 0.80
        }
    
    def extract(self, pdf_path: str, doc_profile: DocumentProfile) -> ExtractedDocument:
        """
        Main extraction method for Strategy B.
        
        Args:
            pdf_path: Path to PDF file
            doc_profile: DocumentProfile from Triage Agent
        
        Returns:
            ExtractedDocument with structured content
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Choose extraction method based on availability
        if self._docling_available:
            result = self._extract_with_docling(pdf_path)
            extraction_method = "docling"
        else:
            # Fallback to pdfplumber with layout heuristics
            result = self._extract_with_fallback(pdf_path)
            extraction_method = "pdfplumber_layout"
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create ExtractedDocument
        extracted_doc = ExtractedDocument(
            doc_id=doc_profile.doc_id,
            filename=doc_profile.filename,
            extraction_id=f"ext_layout_{doc_profile.doc_id}_{int(time.time())}",
            extraction_strategy="layout_aware",
            extraction_started_at=datetime.utcnow(),
            extraction_completed_at=datetime.utcnow(),
            processing_time_seconds=round(processing_time, 2),
            overall_confidence=result.get('confidence', 0.80),
            pages_processed=doc_profile.page_count,
            pages_with_content=doc_profile.page_count,
            pages_failed=[],
            blocks=result.get('blocks', []),
            tables=result.get('tables', []),
            figures=result.get('figures', []),
            reading_order=result.get('reading_order', []),
            cost_estimate_usd=0.0,  # Local processing, no API cost
            tokens_used=0,
            escalation_count=0,
            escalation_log=[]
        )
        
        return extracted_doc
    
    def should_escalate(self, extracted_doc: ExtractedDocument) -> tuple[bool, Optional[str]]:
        """
        Determine if extraction should escalate to Strategy C (VLM).
        
        Returns:
            (should_escalate, reason)
        """
        if extracted_doc.overall_confidence >= self.confidence_threshold:
            return False, None
        
        reason = f"Layout-aware confidence ({extracted_doc.overall_confidence}) below threshold ({self.confidence_threshold})"
        return True, reason