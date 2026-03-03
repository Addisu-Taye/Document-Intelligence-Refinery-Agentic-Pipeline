# src/strategies/fast_text.py
"""
Strategy A: Fast Text Extraction

Uses pdfplumber/pymupdf for native digital PDFs with simple layouts.
Implements multi-signal confidence scoring to detect when escalation is needed.

Triggers when:
- origin_type = native_digital
- layout_complexity = single_column
- character density > threshold
- image ratio < threshold
"""

import pdfplumber
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field

from src.models.extracted_document import ExtractedDocument, ExtractedBlock, BoundingBox
from src.models.document_profile import DocumentProfile


class ConfidenceMetrics(BaseModel):
    """Multi-signal confidence metrics for fast text extraction."""
    char_count: int = Field(..., description="Total characters extracted")
    char_density: float = Field(..., description="Characters per point²")
    image_ratio: float = Field(..., description="Image area / page area")
    has_font_metadata: bool = Field(..., description="Font metadata present")
    table_count: int = Field(..., description="Tables detected")
    whitespace_ratio: float = Field(..., description="Whitespace to content ratio")
    
    # Computed confidence score (0.0 - 1.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Escalation recommendation
    should_escalate: bool = Field(..., description="True if confidence < threshold")
    escalation_reason: Optional[str] = Field(default=None)


class FastTextExtractor:
    """
    Strategy A: Fast text extraction using pdfplumber.
    
    This strategy is low-cost ($0.00) but only works on native digital PDFs
    with simple layouts. Implements multi-signal confidence scoring to detect
    when escalation to Strategy B is needed.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize with configuration from extraction_rules.yaml.
        
        Args:
            config: Configuration dict with confidence thresholds
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get('fast_text', {}).get('min_confidence_to_proceed', 0.75)
        
        # Weights for confidence calculation
        self.weights = self.config.get('fast_text', {}).get('weights', {
            'char_density': 0.4,
            'image_ratio': 0.3,
            'table_structure_preserved': 0.2,
            'font_metadata_present': 0.1
        })
    
    def _compute_bounding_box(self, char: dict) -> Optional[BoundingBox]:
        """Compute bounding box from pdfplumber char dict."""
        try:
            return BoundingBox(
                x0=char.get('x0', 0.0),
                y0=char.get('top', 0.0),
                x1=char.get('x1', 0.0),
                y1=char.get('bottom', 0.0)
            )
        except Exception:
            return None
    
    def _compute_char_density(self, page: pdfplumber.page.Page) -> float:
        """Compute character density (chars per point²)."""
        page_area = page.width * page.height
        char_count = len(page.chars) if hasattr(page, 'chars') else 0
        
        if page_area <= 0:
            return 0.0
        
        return char_count / page_area
    
    def _compute_image_ratio(self, page: pdfplumber.page.Page) -> float:
        """Compute image area ratio (image area / page area)."""
        page_area = page.width * page.height
        images = page.images if hasattr(page, 'images') else []
        
        if page_area <= 0 or not images:
            return 0.0
        
        image_area = sum(
            img.get('width', 0) * img.get('height', 0) 
            for img in images
        )
        
        return image_area / page_area
    
    def _compute_whitespace_ratio(self, text: str) -> float:
        """Compute whitespace to content ratio."""
        if not text:
            return 1.0
        
        whitespace_count = sum(1 for c in text if c.isspace())
        total_count = len(text)
        
        return whitespace_count / total_count if total_count > 0 else 1.0
    
    def _compute_confidence(self, page: pdfplumber.page.Page) -> ConfidenceMetrics:
        """
        Compute multi-signal confidence score for a page.
        
        Signals:
        1. Character density (high = good for fast text)
        2. Image ratio (low = good for fast text)
        3. Font metadata presence (present = good)
        4. Table structure (preserved = good)
        5. Whitespace ratio (reasonable = good)
        """
        # Extract metrics
        char_count = len(page.chars) if hasattr(page, 'chars') else 0
        char_density = self._compute_char_density(page)
        image_ratio = self._compute_image_ratio(page)
        
        # Font metadata check
        fonts_found = set()
        if hasattr(page, 'chars') and page.chars:
            for char in page.chars[:100]:
                font = char.get('fontname', '')
                if font:
                    fonts_found.add(font)
        has_font_metadata = len(fonts_found) > 0
        
        # Table count
        try:
            tables = page.find_tables()
            table_count = len(tables)
        except Exception:
            table_count = 0
        
        # Text extraction for whitespace analysis
        text = page.extract_text() or ""
        whitespace_ratio = self._compute_whitespace_ratio(text)
        
        # Compute individual signal scores (0.0 - 1.0)
        # Char density score: higher is better (threshold: 0.001)
        char_density_score = min(char_density / 0.001, 1.0)
        
        # Image ratio score: lower is better (threshold: 0.5)
        image_ratio_score = max(1.0 - (image_ratio / 0.5), 0.0)
        
        # Font metadata score: binary
        font_score = 1.0 if has_font_metadata else 0.3
        
        # Table structure score: tables detected is good (we can extract them)
        table_score = min(table_count / 5.0, 1.0) if table_count > 0 else 0.7
        
        # Compute weighted overall confidence
        overall_confidence = (
            self.weights.get('char_density', 0.4) * char_density_score +
            self.weights.get('image_ratio', 0.3) * image_ratio_score +
            self.weights.get('table_structure_preserved', 0.2) * table_score +
            self.weights.get('font_metadata_present', 0.1) * font_score
        )
        
        # Determine if escalation is needed
        should_escalate = overall_confidence < self.confidence_threshold
        escalation_reason = None
        
        if should_escalate:
            reasons = []
            if char_density_score < 0.5:
                reasons.append(f"low char density ({char_density:.6f})")
            if image_ratio_score < 0.5:
                reasons.append(f"high image ratio ({image_ratio:.2f})")
            if not has_font_metadata:
                reasons.append("no font metadata")
            
            escalation_reason = "; ".join(reasons) if reasons else "overall confidence below threshold"
        
        return ConfidenceMetrics(
            char_count=char_count,
            char_density=char_density,
            image_ratio=image_ratio,
            has_font_metadata=has_font_metadata,
            table_count=table_count,
            whitespace_ratio=whitespace_ratio,
            overall_confidence=round(overall_confidence, 3),
            should_escalate=should_escalate,
            escalation_reason=escalation_reason
        )
    
    def _extract_blocks(self, page: pdfplumber.page.Page, page_num: int) -> list[ExtractedBlock]:
        """Extract text blocks from a page with bounding boxes."""
        blocks = []
        
        # Extract text by line for better structure
        lines = page.extract_text_lines() if hasattr(page, 'extract_text_lines') else []
        
        if lines:
            for i, line in enumerate(lines):
                text = line.get('text', '').strip()
                if not text:
                    continue
                
                # Compute bounding box from line chars
                chars = line.get('chars', [])
                if chars:
                    bbox = BoundingBox(
                        x0=min(c.get('x0', 0) for c in chars),
                        y0=min(c.get('top', 0) for c in chars),
                        x1=max(c.get('x1', 0) for c in chars),
                        y1=max(c.get('bottom', 0) for c in chars)
                    )
                else:
                    bbox = None
                
                block = ExtractedBlock(
                    block_id=f"blk_p{page_num}_l{i}_{hashlib.sha256(text.encode()).hexdigest()[:8]}",
                    content=text,
                    block_type="text",
                    page=page_num,
                    bbox=bbox,
                    confidence=0.85  # Default confidence for fast text
                )
                blocks.append(block)
        else:
            # Fallback: extract full page text
            text = page.extract_text() or ""
            if text.strip():
                block = ExtractedBlock(
                    block_id=f"blk_p{page_num}_full",
                    content=text,
                    block_type="text",
                    page=page_num,
                    bbox=None,
                    confidence=0.70  # Lower confidence for full-page dump
                )
                blocks.append(block)
        
        return blocks
    
    def _extract_tables(self, page: pdfplumber.page.Page, page_num: int) -> list[dict]:
        """Extract tables as structured objects."""
        tables = []
        
        try:
            pdf_tables = page.find_tables()
            
            for i, table in enumerate(pdf_tables):
                table_data = table.extract()
                if not table_data or len(table_data) < 2:
                    continue
                
                # First row as headers
                headers = [str(h) if h else f"col_{j}" for j, h in enumerate(table_data[0])]
                rows = [[str(cell) if cell else "" for cell in row] for row in table_data[1:]]
                
                # Compute bounding box
                bbox = None
                if hasattr(table, 'bbox') and table.bbox:
                    bbox = {
                        'x0': table.bbox[0],
                        'y0': table.bbox[1],
                        'x1': table.bbox[2],
                        'y1': table.bbox[3]
                    }
                
                table_obj = {
                    'table_id': f"tbl_p{page_num}_t{i}",
                    'page': page_num,
                    'bbox': bbox,
                    'headers': headers,
                    'rows': rows,
                    'caption': None,  # Would need layout analysis for captions
                    'row_count': len(rows),
                    'column_count': len(headers)
                }
                tables.append(table_obj)
        
        except Exception as e:
            pass  # Table extraction failed, continue with text
        
        return tables
    
    def extract(self, pdf_path: str, doc_profile: DocumentProfile) -> ExtractedDocument:
        """
        Main extraction method for Strategy A.
        
        Args:
            pdf_path: Path to PDF file
            doc_profile: DocumentProfile from Triage Agent
        
        Returns:
            ExtractedDocument with blocks, tables, figures, and confidence metrics
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        blocks = []
        tables = []
        figures = []
        pages_failed = []
        reading_order = []
        page_confidences = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_num = page.page_number
                
                try:
                    # Compute confidence for this page
                    confidence_metrics = self._compute_confidence(page)
                    page_confidences.append(confidence_metrics.overall_confidence)
                    
                    # Extract text blocks
                    page_blocks = self._extract_blocks(page, page_num)
                    for block in page_blocks:
                        blocks.append(block)
                        reading_order.append(block.block_id)
                    
                    # Extract tables
                    page_tables = self._extract_tables(page, page_num)
                    tables.extend(page_tables)
                    
                except Exception as e:
                    pages_failed.append(page_num)
        
        # Compute overall confidence (median across pages)
        overall_confidence = sorted(page_confidences)[len(page_confidences) // 2] if page_confidences else 0.0
        
        # Check if escalation is needed
        should_escalate = overall_confidence < self.confidence_threshold
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create ExtractedDocument
        extracted_doc = ExtractedDocument(
            doc_id=doc_profile.doc_id,
            filename=doc_profile.filename,
            extraction_id=f"ext_fast_{doc_profile.doc_id}_{int(time.time())}",
            extraction_strategy="fast_text",
            extraction_started_at=datetime.utcnow(),
            extraction_completed_at=datetime.utcnow(),
            processing_time_seconds=round(processing_time, 2),
            overall_confidence=round(overall_confidence, 3),
            pages_processed=len(pdf.pages),
            pages_with_content=len(pdf.pages) - len(pages_failed),
            pages_failed=pages_failed,
            blocks=blocks,
            tables=tables,
            figures=figures,
            reading_order=reading_order,
            cost_estimate_usd=0.0,  # Strategy A is free
            tokens_used=0,
            escalation_count=0,
            escalation_log=[]
        )
        
        return extracted_doc
    
    def should_escalate(self, extracted_doc: ExtractedDocument) -> tuple[bool, Optional[str]]:
        """
        Determine if extraction should escalate to Strategy B.
        
        Returns:
            (should_escalate, reason)
        """
        if extracted_doc.overall_confidence >= self.confidence_threshold:
            return False, None
        
        reason = f"Fast text confidence ({extracted_doc.overall_confidence}) below threshold ({self.confidence_threshold})"
        return True, reason