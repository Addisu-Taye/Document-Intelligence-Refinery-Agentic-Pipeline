# src/agents/chunker.py — FIXED
"""
Semantic Chunker — Merge small blocks to keep financial context intact.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

from src.models.extracted_document import ExtractedDocument, Block
from src.models.ldu import LDU

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('.refinery/chunker.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SemanticChunker:
    """Chunk documents semantically while preserving financial context."""
    
    # Constitutional rules for valid LDUs
    MIN_TOKENS = 20
    MAX_TOKENS = 400  # ← Increased from default for better context
    OVERLAP_TOKENS = 80
    
    def chunk(self, doc: ExtractedDocument) -> 'ChunkResult':
        """Chunk document into LDUs with merged small blocks."""
        # FIX: Merge small blocks before chunking
        merged_blocks = self._merge_small_blocks(doc.blocks)
        
        ldus = []
        for block in merged_blocks:
            if block.content and block.content.strip():
                ldu = self._block_to_ldu(block)
                if self._validate_ldu(ldu):
                    ldus.append(ldu)
        
        # Apply overlap for adjacent LDUs
        ldus = self._apply_overlap(ldus)
        
        result = ChunkResult(
            ldus=ldus,
            total_ldus=len(ldus),
            validation_passed=all(self._validate_ldu(l) for l in ldus),
            rules_enforced=5
        )
        
        logger.info(f"✓ Chunked: {len(doc.blocks)} blocks → {len(ldus)} LDUs")
        return result
    
    def _merge_small_blocks(self, blocks: List[Block]) -> List[Block]:
        """
        Merge small text blocks to keep financial context intact.
        
        Problem: "net profit" and "ETB 5.4 billion" may be in separate blocks.
        Solution: Merge blocks <120 chars with adjacent content.
        """
        if not blocks:
            return blocks
        
        merged = []
        buffer = None
        
        for block in blocks:
            content = block.content.strip() if block.content else ''
            
            # Skip empty blocks
            if not content:
                continue
            
            # If buffer exists and current block is small, merge
            if buffer and len(content) < 120:
                buffer.content = buffer.content + " " + content
                # Merge page refs
                if block.page_refs and buffer.page_refs:
                    if isinstance(buffer.page_refs, list) and isinstance(block.page_refs, list):
                        buffer.page_refs = list(set(buffer.page_refs + block.page_refs))
            # If buffer exists and current block is large, save buffer and start new
            elif buffer:
                merged.append(buffer)
                buffer = block if len(content) >= 120 else None
                if buffer and len(content) < 120:
                    buffer.content = content
            # If no buffer and current block is small, start buffer
            elif len(content) < 120:
                buffer = Block(
                    content=content,
                    block_type=block.block_type,
                    page_refs=block.page_refs,
                    bbox=block.bbox
                )
            # If no buffer and current block is large, add directly
            else:
                merged.append(block)
        
        # Don't forget the last buffer
        if buffer:
            merged.append(buffer)
        
        logger.debug(f"Merged {len(blocks)} blocks → {len(merged)} blocks")
        return merged
    
    def _block_to_ldu(self, block: Block) -> LDU:
        """Convert block to LDU with complete metadata."""
        content = block.content.strip() if block.content else ''
        
        return LDU(
            ldu_id=f"ldu_{hash(content) % 100000}",
            content=content,
            chunk_type=block.block_type or "text",
            page_refs=block.page_refs or [],
            section_path=[],  # Would need to extract from document structure
            token_count=len(content.split()),
            extraction_strategy="layout_aware",
            bbox=block.bbox
        )
    
    def _validate_ldu(self, ldu: LDU) -> bool:
        """Validate LDU against constitutional rules."""
        if not ldu.content or not ldu.content.strip():
            return False
        if ldu.token_count < self.MIN_TOKENS:
            return False
        if ldu.token_count > self.MAX_TOKENS:
            return False
        return True
    
    def _apply_overlap(self, ldus: List[LDU]) -> List[LDU]:
        """Apply token overlap between adjacent LDUs for better retrieval."""
        if len(ldus) < 2:
            return ldus
        
        result = []
        for i, ldu in enumerate(ldus):
            # Add overlap from previous LDU
            if i > 0 and result:
                prev = result[-1]
                prev_tokens = prev.content.split()
                overlap_start = max(0, len(prev_tokens) - self.OVERLAP_TOKENS)
                overlap_text = " ".join(prev_tokens[overlap_start:])
                if overlap_text and overlap_text not in ldu.content:
                    ldu.content = overlap_text + " " + ldu.content
                    ldu.token_count = len(ldu.content.split())
            
            result.append(ldu)
        
        return result


class ChunkResult(BaseModel):
    """Result of semantic chunking."""
    ldus: List[LDU]
    total_ldus: int
    validation_passed: bool
    rules_enforced: int


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    """Test chunker with block merging."""
    import sys
    sys.path.insert(0, '.')
    
    from src.agents.extractor import ExtractionRouter
    from src.agents.triage import TriageAgent
    from pathlib import Path
    
    print("🔄 Testing chunker with block merging...")
    
    triage = TriageAgent()
    router = ExtractionRouter()
    chunker = SemanticChunker()
    
    # Test with first PDF in corpus
    corpus_dir = Path("corpus")
    if not corpus_dir.exists():
        print("❌ corpus/ directory not found")
        sys.exit(1)
    
    pdf = list(corpus_dir.glob("*.pdf"))[0]
    print(f"Testing with: {pdf.name}")
    
    profile = triage.profile_document(str(pdf))
    extraction = router.extract(str(pdf), profile)
    
    print(f"Original blocks: {len(extraction.extracted_document.blocks)}")
    
    result = chunker.chunk(extraction.extracted_document)
    
    print(f"Merged LDUs: {result.total_ldus}")
    print(f"Validation: {'✓ Passed' if result.validation_passed else '✗ Failed'}")
    
    # Show sample LDUs
    print("\nSample LDUs:")
    for i, ldu in enumerate(result.ldus[:3]):
        print(f"\n[{i+1}] {ldu.chunk_type} ({ldu.token_count} tokens)")
        print(f"    {ldu.content[:150]}...")