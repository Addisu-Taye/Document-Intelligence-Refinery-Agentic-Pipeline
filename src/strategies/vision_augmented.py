# src/strategies/vision_augmented.py
"""
Strategy C: Vision-Augmented Extraction

Uses VLM (GPT-4o-mini, Gemini Flash) via OpenRouter for scanned documents
or when Strategies A/B fail. Highest quality but expensive.

Triggers when:
- origin_type = scanned_image
- Strategy A/B confidence < threshold
- handwriting detected

Mandatory: Budget guard to prevent cost overrun.
"""

import time
import os
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field

from src.models.extracted_document import ExtractedDocument, ExtractedBlock, BoundingBox
from src.models.document_profile import DocumentProfile


class BudgetGuard(BaseModel):
    """
    Budget tracking for VLM usage.
    
    Prevents cost overrun by tracking token spend per document.
    """
    max_cost_usd: float = Field(default=2.00, description="Maximum cost per document")
    max_tokens: int = Field(default=50000, description="Maximum tokens per document")
    rate_per_1k_tokens: float = Field(default=0.0001, description="Cost per 1K tokens")
    
    # Tracking
    tokens_used: int = Field(default=0, ge=0)
    cost_accumulated: float = Field(default=0.0, ge=0.0)
    
    def can_proceed(self, estimated_tokens: int) -> tuple[bool, Optional[str]]:
        """Check if request is within budget."""
        estimated_cost = (estimated_tokens / 1000) * self.rate_per_1k_tokens
        
        if self.tokens_used + estimated_tokens > self.max_tokens:
            return False, f"Token limit exceeded ({self.tokens_used + estimated_tokens} > {self.max_tokens})"
        
        if self.cost_accumulated + estimated_cost >= self.max_cost_usd:
            return False, f"Budget exceeded (${self.cost_accumulated + estimated_cost:.4f} > ${self.max_cost_usd})"
        
        return True, None
    
    def record_usage(self, tokens_used: int):
        """Record token usage after API call."""
        cost = (tokens_used / 1000) * self.rate_per_1k_tokens
        self.tokens_used += tokens_used
        self.cost_accumulated += cost


class VisionExtractor:
    """
    Strategy C: Vision-augmented extraction using VLM via OpenRouter.
    
    This strategy handles scanned documents, handwritten content, and cases
    where Strategies A/B fail. It's expensive but highest quality.
    
    Note: This is a stub implementation. In production, you would integrate
    actual OpenRouter API calls here.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize with configuration from extraction_rules.yaml.
        
        Args:
            config: Configuration dict with cost limits and thresholds
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get('vision_augmented', {}).get('min_confidence_to_proceed', 0.90)
        
        # Initialize budget guard
        cost_limits = self.config.get('cost_limits', {}).get('vision_augmented', {})
        self.budget = BudgetGuard(
            max_cost_usd=cost_limits.get('max_cost_usd', 2.00),
            max_tokens=cost_limits.get('max_tokens_per_doc', 50000),
            rate_per_1k_tokens=cost_limits.get('rate_per_1k_tokens', 0.0001)
        )
        
        # API configuration
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.model = os.getenv('VLM_MODEL', 'openai/gpt-4o-mini')
        self._api_available = bool(self.api_key)
    
    def _encode_image_to_base64(self, image_path: Path) -> str:
        """Encode image to base64 for API transmission."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _extract_with_vlm(self, pdf_path: Path, page_nums: list[int]) -> dict:
        """
        Extract content using VLM API.
        
        In production, this would:
        1. Convert PDF pages to images
        2. Send to OpenRouter with structured extraction prompt
        3. Parse JSON response into ExtractedDocument format
        """
        # Stub implementation - would use actual OpenRouter API in production
        # Example API call structure:
        #
        # import httpx
        # response = httpx.post(
        #     "https://openrouter.ai/api/v1/chat/completions",
        #     headers={
        #         "Authorization": f"Bearer {self.api_key}",
        #         "HTTP-Referer": "https://your-app.com",
        #         "X-Title": "Document Refinery"
        #     },
        #     json={
        #         "model": self.model,
        #         "messages": [
        #             {
        #                 "role": "user",
        #                 "content": [
        #                     {"type": "text", "text": EXTRACTION_PROMPT},
        #                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        #                 ]
        #             }
        #         ]
        #     }
        # )
        
        # For now, return stub result
        return {
            'blocks': [],
            'tables': [],
            'figures': [],
            'reading_order': [],
            'confidence': 0.90,  # VLM typically high confidence
            'tokens_used': 1000,  # Estimated
            'extraction_method': 'vlm_stub'
        }
    
    def _extract_with_fallback(self, pdf_path: Path) -> dict:
        """
        Fallback when VLM API is not available.
        
        Uses pdfplumber with OCR-like heuristics.
        """
        import pdfplumber
        import hashlib
        
        blocks = []
        tables = []
        reading_order = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_num = page.page_number
                text = page.extract_text() or ""
                
                if text.strip():
                    block = ExtractedBlock(
                        block_id=f"blk_vlm_p{page_num}",
                        content=text,
                        block_type="text",
                        page=page_num,
                        bbox=None,
                        confidence=0.75  # Lower confidence for fallback
                    )
                    blocks.append(block)
                    reading_order.append(block.block_id)
        
        return {
            'blocks': blocks,
            'tables': tables,
            'figures': [],
            'reading_order': reading_order,
            'confidence': 0.75,
            'tokens_used': 0,
            'extraction_method': 'pdfplumber_fallback'
        }
    
    def extract(self, pdf_path: str, doc_profile: DocumentProfile) -> ExtractedDocument:
        """
        Main extraction method for Strategy C.
        
        Args:
            pdf_path: Path to PDF file
            doc_profile: DocumentProfile from Triage Agent
        
        Returns:
            ExtractedDocument with VLM-extracted content
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Check budget before proceeding
        estimated_tokens = doc_profile.page_count * 500  # Rough estimate
        can_proceed, budget_error = self.budget.can_proceed(estimated_tokens)
        
        if not can_proceed:
            # Fallback to local extraction
            result = self._extract_with_fallback(pdf_path)
            extraction_method = "fallback"
        elif self._api_available:
            # Use VLM API
            result = self._extract_with_vlm(pdf_path, list(range(1, min(6, doc_profile.page_count + 1))))
            extraction_method = "vlm"
            self.budget.record_usage(result.get('tokens_used', 0))
        else:
            # No API key, use fallback
            result = self._extract_with_fallback(pdf_path)
            extraction_method = "fallback"
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create ExtractedDocument
        extracted_doc = ExtractedDocument(
            doc_id=doc_profile.doc_id,
            filename=doc_profile.filename,
            extraction_id=f"ext_vlm_{doc_profile.doc_id}_{int(time.time())}",
            extraction_strategy="vision_augmented",
            extraction_started_at=datetime.utcnow(),
            extraction_completed_at=datetime.utcnow(),
            processing_time_seconds=round(processing_time, 2),
            overall_confidence=result.get('confidence', 0.75),
            pages_processed=doc_profile.page_count,
            pages_with_content=doc_profile.page_count,
            pages_failed=[],
            blocks=result.get('blocks', []),
            tables=result.get('tables', []),
            figures=result.get('figures', []),
            reading_order=result.get('reading_order', []),
            cost_estimate_usd=self.budget.cost_accumulated,
            tokens_used=self.budget.tokens_used,
            escalation_count=0,
            escalation_log=[]
        )
        
        return extracted_doc
    
    def should_escalate(self, extracted_doc: ExtractedDocument) -> tuple[bool, Optional[str]]:
        """
        Determine if extraction should escalate (Strategy C is final).
        
        Returns:
            (should_escalate, reason) - Always False for Strategy C
        """
        # Strategy C is the final escalation - no further escalation possible
        if extracted_doc.overall_confidence < self.confidence_threshold:
            return False, f"VLM confidence ({extracted_doc.overall_confidence}) below ideal ({self.confidence_threshold}), but no further escalation available"
        
        return False, None
    
    def get_budget_status(self) -> dict:
        """Get current budget status."""
        return {
            'tokens_used': self.budget.tokens_used,
            'max_tokens': self.budget.max_tokens,
            'cost_accumulated': self.budget.cost_accumulated,
            'max_cost_usd': self.budget.max_cost_usd,
            'remaining_budget': self.budget.max_cost_usd - self.budget.cost_accumulated
        }
