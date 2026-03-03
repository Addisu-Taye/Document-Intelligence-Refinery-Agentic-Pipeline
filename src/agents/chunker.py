# src/agents/chunker.py
"""
Semantic Chunking Engine

Transforms raw ExtractedDocument into Logical Document Units (LDUs)
that preserve semantic coherence and structural context.

Enforces the 5 Chunking Rules (the "Constitution"):
1. Table cells never split from header rows
2. Figure captions stored as parent metadata
3. Numbered lists kept atomic (unless > max_tokens)
4. Section headers propagated to child chunks
5. Cross-references resolved as chunk relationships
"""

import hashlib
import re
from typing import Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from src.models.ldu import LDU, BoundingBox
from src.models.extracted_document import ExtractedDocument, ExtractedBlock
from src.agents.validator import ChunkValidator


class ChunkingConfig(BaseModel):
    """Configuration for chunking behavior."""
    max_tokens_per_ldu: int = 512
    min_tokens_per_ldu: int = 50
    preserve_table_structure: bool = True
    keep_captions_with_figures: bool = True
    section_header_as_parent_metadata: bool = True
    numbered_list_atomic_unit: bool = True
    cross_reference_resolution: bool = True


class ChunkingResult(BaseModel):
    """Output of the Chunking Engine."""
    ldus: list[LDU]
    total_ldus: int
    chunking_config: ChunkingConfig
    validation_passed: bool
    validation_errors: list[str]
    processing_time_seconds: float
    document_id: str


class SemanticChunker:
    """
    Semantic Chunking Engine that converts ExtractedDocument to LDUs.
    
    Enforces the 5 Chunking Rules to prevent context poverty in RAG.
    """
    
    # Pattern for numbered lists (1., 2., (1), (a), etc.)
    LIST_PATTERN = re.compile(r'^(\d+\.|\(\d+\)|[a-z]\.|[a-z]\))\s+')
    
    # Pattern for cross-references (e.g., "see Table 3", "Figure 2.1")
    CROSS_REF_PATTERN = re.compile(r'(see\s+)?(Table|Figure|Section|Chapter)\s+(\d+\.?\d*)', re.IGNORECASE)
    
    def __init__(self, config: ChunkingConfig = None):
        """
        Initialize chunker with configuration.
        
        Args:
            config: ChunkingConfig with token limits and rule flags
        """
        self.config = config or ChunkingConfig()
        self.validator = ChunkValidator()
    
    def _compute_content_hash(self, content: str, page_refs: list[int], bbox: Optional[BoundingBox]) -> str:
        """Generate deterministic SHA-256 hash for provenance verification."""
        bbox_str = f"{bbox.to_tuple()}" if bbox else "None"
        data = f"{content}|{page_refs}|{bbox_str}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: 1 token ≈ 4 characters for English)."""
        if not text:
            return 0
        return len(text) // 4
    
    def _extract_section_hierarchy(self, blocks: list[ExtractedBlock]) -> dict[int, list[str]]:
        """
        Extract section hierarchy from blocks.
        
        Returns:
            Dict mapping page numbers to section path lists.
        """
        hierarchy = {}
        current_path = []
        
        for block in blocks:
            if block.block_type == "header":
                # Update section path
                header_text = block.content.strip()
                # Simple heuristic: short headers are section titles
                if len(header_text) < 100:
                    current_path.append(header_text)
            
            page = block.page
            if page not in hierarchy:
                hierarchy[page] = current_path.copy()
        
        return hierarchy
    
    def _chunk_table(self, table: dict, page: int, section_path: list[str], extraction_strategy: str) -> LDU:
        """
        Create LDU from table (Rule 1: cells never split from headers).
        
        The entire table is kept as one atomic unit to preserve structure.
        """
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        # Serialize table to text while preserving structure
        header_row = " | ".join(headers)
        data_rows = "\n".join(" | ".join(str(cell) for cell in row) for row in rows)
        content = f"Headers: {header_row}\nData:\n{data_rows}"
        
        # Compute bounding box from table metadata
        bbox = None
        if table.get('bbox'):
            bbox = BoundingBox(
                x0=table['bbox'].get('x0', 0),
                y0=table['bbox'].get('y0', 0),
                x1=table['bbox'].get('x1', 0),
                y1=table['bbox'].get('y1', 0)
            )
        
        token_count = self._estimate_tokens(content)
        
        return LDU(
            ldu_id=f"ldu_tbl_{table.get('table_id', 'unknown')}_{hashlib.sha256(content[:50].encode()).hexdigest()[:8]}",
            content=content,
            chunk_type="table",
            page_refs=[page],
            bounding_box=bbox,
            parent_section=section_path[-1] if section_path else None,
            section_path=section_path,
            token_count=token_count,
            word_count=len(content.split()),
            table_headers=headers,
            table_row_count=len(rows),
            table_data=[{headers[i]: row[i] for i in range(len(headers))} for row in rows] if headers and rows else None,
            content_hash=self._compute_content_hash(content, [page], bbox),
            extraction_strategy=extraction_strategy,
            extraction_confidence=0.85
        )
    
    def _chunk_figure(self, figure: dict, page: int, section_path: list[str], extraction_strategy: str) -> LDU:
        """
        Create LDU from figure (Rule 2: caption stored as parent metadata).
        """
        caption = figure.get('caption', 'No caption available')
        figure_type = figure.get('type', 'unknown')
        
        # Content includes caption for semantic coherence
        content = f"[{figure_type}] {caption}"
        
        bbox = None
        if figure.get('bbox'):
            bbox = BoundingBox(
                x0=figure['bbox'].get('x0', 0),
                y0=figure['bbox'].get('y0', 0),
                x1=figure['bbox'].get('x1', 0),
                y1=figure['bbox'].get('y1', 0)
            )
        
        token_count = self._estimate_tokens(content)
        
        return LDU(
            ldu_id=f"ldu_fig_{figure.get('figure_id', 'unknown')}_{hashlib.sha256(content[:50].encode()).hexdigest()[:8]}",
            content=content,
            chunk_type="figure",
            page_refs=[page],
            bounding_box=bbox,
            parent_section=section_path[-1] if section_path else None,
            section_path=section_path,
            token_count=token_count,
            word_count=len(content.split()),
            figure_caption=caption,
            figure_type=figure_type,
            content_hash=self._compute_content_hash(content, [page], bbox),
            extraction_strategy=extraction_strategy,
            extraction_confidence=0.80
        )
    
    def _chunk_text_block(self, block: ExtractedBlock, section_path: list[str], extraction_strategy: str) -> LDU:
        """
        Create LDU from text block with section header propagation (Rule 4).
        """
        content = block.content.strip()
        if not content:
            return None
        
        bbox = None
        if block.bbox:
            bbox = BoundingBox(
                x0=block.bbox.x0,
                y0=block.bbox.y0,
                x1=block.bbox.x1,
                y1=block.bbox.y1
            )
        
        token_count = self._estimate_tokens(content)
        
        # Determine chunk type
        if self.LIST_PATTERN.match(content):
            chunk_type = "list"
        else:
            chunk_type = "text_block"
        
        return LDU(
            ldu_id=f"ldu_txt_{block.block_id}_{hashlib.sha256(content[:50].encode()).hexdigest()[:8]}",
            content=content,
            chunk_type=chunk_type,
            page_refs=[block.page],
            bounding_box=bbox,
            parent_section=section_path[-1] if section_path else None,
            section_path=section_path,
            token_count=token_count,
            word_count=len(content.split()),
            content_hash=self._compute_content_hash(content, [block.page], bbox),
            extraction_strategy=extraction_strategy,
            extraction_confidence=block.confidence
        )
    
    def _resolve_cross_references(self, ldus: list[LDU]) -> list[LDU]:
        """
        Resolve cross-references between LDUs (Rule 5).
        
        Scans for "see Table 3" patterns and creates relationships.
        """
        if not self.config.cross_reference_resolution:
            return ldus
        
        # Build lookup tables
        table_ldus = {i: ldu for i, ldu in enumerate(ldus) if ldu.chunk_type == "table"}
        figure_ldus = {i: ldu for i, ldu in enumerate(ldus) if ldu.chunk_type == "figure"}
        
        # Scan for cross-references
        for i, ldu in enumerate(ldus):
            refs = self.CROSS_REF_PATTERN.findall(ldu.content)
            
            for ref in refs:
                ref_type = ref[1].lower()  # table, figure, section
                ref_number = ref[2]
                
                # Find matching LDU
                if ref_type == "table":
                    for j, table_ldu in table_ldus.items():
                        if j != i and ref_number in table_ldu.content:
                            if table_ldu.ldu_id not in ldu.cross_references:
                                ldu.cross_references.append(table_ldu.ldu_id)
                            if ldu.ldu_id not in table_ldu.referenced_by:
                                table_ldu.referenced_by.append(ldu.ldu_id)
                
                elif ref_type == "figure":
                    for j, fig_ldu in figure_ldus.items():
                        if j != i and ref_number in fig_ldu.content:
                            if fig_ldu.ldu_id not in ldu.cross_references:
                                ldu.cross_references.append(fig_ldu.ldu_id)
                            if ldu.ldu_id not in fig_ldu.referenced_by:
                                fig_ldu.referenced_by.append(ldu.ldu_id)
        
        return ldus
    
    def chunk(self, extracted_doc: ExtractedDocument) -> ChunkingResult:
        """
        Main chunking method: Convert ExtractedDocument to LDUs.
        
        Args:
            extracted_doc: Normalized extraction output from any strategy
        
        Returns:
            ChunkingResult with LDUs and validation status
        """
        import time
        start_time = time.time()
        
        ldus = []
        
        # Extract section hierarchy
        section_hierarchy = self._extract_section_hierarchy(extracted_doc.blocks)
        
        # Process text blocks
        for block in extracted_doc.blocks:
            section_path = section_hierarchy.get(block.page, [])
            ldu = self._chunk_text_block(block, section_path, extracted_doc.extraction_strategy)
            if ldu:
                ldus.append(ldu)
        
        # Process tables (Rule 1: atomic units)
        for table in extracted_doc.tables:
            page = table.get('page', 1)
            section_path = section_hierarchy.get(page, [])
            ldu = self._chunk_table(table, page, section_path, extracted_doc.extraction_strategy)
            ldus.append(ldu)
        
        # Process figures (Rule 2: caption as metadata)
        for figure in extracted_doc.figures:
            page = figure.get('page', 1)
            section_path = section_hierarchy.get(page, [])
            ldu = self._chunk_figure(figure, page, section_path, extracted_doc.extraction_strategy)
            ldus.append(ldu)
        
        # Resolve cross-references (Rule 5)
        ldus = self._resolve_cross_references(ldus)
        
        # Validate all LDUs against the 5 rules
        validation_errors = []
        for ldu in ldus:
            is_valid, errors = self.validator.validate(ldu)
            if not is_valid:
                validation_errors.extend(errors)
        
        end_time = time.time()
        
        return ChunkingResult(
            ldus=ldus,
            total_ldus=len(ldus),
            chunking_config=self.config,
            validation_passed=len(validation_errors) == 0,
            validation_errors=validation_errors,
            processing_time_seconds=round(end_time - start_time, 2),
            document_id=extracted_doc.doc_id
        )


def main():
    """CLI entry point for chunking."""
    import sys
    import json
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.chunker <doc_id>")
        sys.exit(1)
    
    doc_id = sys.argv[1]
    
    # Load extracted document (would come from extraction pipeline)
    # For now, this is a placeholder
    print(f"🔧 Chunking document: {doc_id}")
    print("  (Full integration with extraction pipeline in Task 06)")


if __name__ == "__main__":
    main()
