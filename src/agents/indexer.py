# src/agents/indexer.py
"""
PageIndex Builder - Hierarchical Navigation Tree Generator

Transforms LDUs into a searchable, hierarchical index that enables
LLM-efficient document navigation without full-document embedding search.

This is Deliverable #2 (Final Submission) from the challenge specification.
"""

import re
import time
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional
from collections import defaultdict

from src.models.ldu import LDU
from src.models.page_index import PageIndex, PageIndexNode
from src.models.extracted_document import ExtractedDocument


class PageIndexBuilder:
    """
    Builds hierarchical PageIndex from extracted LDUs.
    
    Features:
    - Automatic section detection from LDU metadata
    - LLM-generated summaries (stub for production integration)
    - Content inventory (tables, figures, entities per section)
    - Keyword extraction for search optimization
    - LDU-to-section mapping for precise retrieval
    """
    
    # Patterns for section header detection
    SECTION_PATTERNS = [
        r'^(Chapter\s+\d+[:\s].*)$',
        r'^(\d+\.\d+[:\s].*)$',
        r'^(\d+[:\s].*)$',
        r'^([A-Z][A-Z\s]+)$',  # ALL CAPS headers
    ]
    
    # Entity extraction patterns (simplified)
    ENTITY_PATTERNS = {
        'currency': r'(ETB|USD|EUR|Birr)\s*[\d,\.]+[BMK]?',
        'percentage': r'[\d\.]+\s*%',
        'date': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
        'organization': r'(?:Commercial\s+)?Bank\s+of\s+Ethiopia|NBE|FTA',
    }
    
    def __init__(self, summary_model: str = "gpt-4o-mini", generate_summaries: bool = False):
        """

    def _normalize_chunk_type(self, chunk_type: str) -> Optional[str]:
        '''Map LDU chunk_type to PageIndex literal values.'''
        mapping = {
            'table': 'tables',
            'figure': 'figures', 
            'text_block': 'text',
            'list': 'lists',
            'equation': 'equations',
            'section_header': None  # Exclude headers from data_types
        }
        return mapping.get(chunk_type)

        Initialize PageIndex builder.
        
        Args:
            summary_model: LLM model for generating section summaries
            generate_summaries: If True, call LLM API for summaries (stub if no API key)
        """
        self.summary_model = summary_model
        self.generate_summaries = generate_summaries
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """Extract top keywords from text using simple frequency analysis."""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could'
        }
        
        # Simple tokenization
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Count frequencies
        freq = defaultdict(int)
        for word in words:
            if word not in stop_words:
                freq[word] += 1
        
        # Return top keywords
        return [word for word, _ in sorted(freq.items(), key=lambda x: -x[1])[:max_keywords]]
    
    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities using pattern matching."""
        entities = []
        
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        # Deduplicate and limit
        return list(dict.fromkeys(entities))[:20]
    
    def _generate_summary_stub(self, ldus: list[LDU], section_title: str) -> str:
        """
        Generate section summary (stub implementation).
        
        In production, this would call an LLM API:
        - Send section content to gpt-4o-mini with prompt
        - Parse JSON response with summary
        - Cache result to avoid regenerating
        
        For now, returns a heuristic summary based on content analysis.
        """
        if not ldus:
            return f"Section '{section_title}' contains no extractable content."
        
        # Combine content for analysis
        content = " ".join(ldu.content for ldu in ldus if ldu.content)
        
        # Heuristic summary generation
        if len(content) < 200:
            return f"{section_title}: {content[:150]}..."
        
        # Extract first sentence as summary starter
        sentences = re.split(r'[.!?]+', content)
        first_sentence = sentences[0].strip() if sentences else content[:200]
        
        # Add content type indicator
        content_types = []
        if any(ldu.chunk_type == "table" for ldu in ldus):
            content_types.append("tables")
        if any(ldu.chunk_type == "figure" for ldu in ldus):
            content_types.append("figures")
        if any(ldu.chunk_type == "list" for ldu in ldus):
            content_types.append("lists")
        
        type_suffix = f" (includes {', '.join(content_types)})" if content_types else ""
        
        return f"{first_sentence}...{type_suffix}"
    
    def _detect_section_hierarchy(self, ldus: list[LDU]) -> dict[str, list[LDU]]:
        """
        Group LDUs by detected section hierarchy.
        
        Uses LDU metadata (section_path, parent_section) to build hierarchy.
        Falls back to header detection if metadata is missing.
        """
        sections = defaultdict(list)
        
        for ldu in ldus:
            # Use explicit section path if available
            if ldu.section_path:
                # Use full path as key for nested sections
                section_key = " > ".join(ldu.section_path)
                sections[section_key].append(ldu)
            elif ldu.parent_section:
                sections[ldu.parent_section].append(ldu)
            elif ldu.chunk_type == "section_header":
                # Treat header as new section
                sections[ldu.content.strip()].append(ldu)
            else:
                # Fallback: group by page ranges
                page_key = f"Pages {ldu.page_refs[0]}-{ldu.page_refs[-1]}"
                sections[page_key].append(ldu)
        
        return dict(sections)
    
    def _build_node(self, section_title: str, ldus: list[LDU], level: int, 
                   parent_id: Optional[str], node_counter: list[int]) -> PageIndexNode:
        """Build a PageIndexNode from section LDUs."""
        node_id = f"node_{node_counter[0]}"
        node_counter[0] += 1
        
        # Page range
        pages = [p for ldu in ldus for p in ldu.page_refs]
        page_start = min(pages) if pages else 1
        page_end = max(pages) if pages else 1
        
        # Content inventory
        table_count = sum(1 for ldu in ldus if ldu.chunk_type == "table")
        figure_count = sum(1 for ldu in ldus if ldu.chunk_type == "figure")
        
        # Normalize chunk types to PageIndex literals (inline)
        chunk_type_map = {
            'table': 'tables', 'figure': 'figures', 'text_block': 'text',
            'list': 'lists', 'equation': 'equations', 'section_header': None
        }
        data_types = []
        for ldu in ldus:
            if ldu.chunk_type != "section_header":
                normalized = chunk_type_map.get(ldu.chunk_type)
                if normalized and normalized not in data_types:
                    data_types.append(normalized)
        
        # Extract keywords and entities from content
        content = " ".join(ldu.content for ldu in ldus if ldu.content)
        keywords = self._extract_keywords(content)
        entities = self._extract_entities(content)
        
        # Generate summary (stub)
        summary = self._generate_summary_stub(ldus, section_title)
        
        # LDU references
        ldu_ids = [ldu.ldu_id for ldu in ldus]
        
        return PageIndexNode(
            node_id=node_id,
            title=section_title,
            level=level,
            parent_node_id=parent_id,
            child_node_ids=[],  # Will be populated after building children
            page_start=page_start,
            page_end=page_end,
            summary=summary,
            summary_model=self.summary_model if self.generate_summaries else "heuristic",
            summary_tokens=len(summary) // 4,  # Rough estimate
            key_entities=entities,
            data_types_present=data_types,
            table_count=table_count,
            figure_count=figure_count,
            keywords=keywords,
            relevance_scores={"financial": 0.8 if "profit" in keywords else 0.3},  # Simple heuristic
            ldu_ids=ldu_ids
        )
    
    def build(self, ldus: list[LDU], doc_id: str, filename: str) -> PageIndex:
        """
        Main method: Build PageIndex from LDUs.
        
        Args:
            ldus: List of Logical Document Units from chunking engine
            doc_id: Document identifier
            filename: Original filename
        
        Returns:
            PageIndex with hierarchical navigation structure
        """
        start_time = time.time()
        
        # Group LDUs by section
        sections = self._detect_section_hierarchy(ldus)
        
        # Build nodes
        nodes = {}
        root_nodes = []
        node_counter = [0]
        
        # Sort sections by title for consistent ordering
        for section_title in sorted(sections.keys()):
            section_ldus = sections[section_title]
            
            # Determine hierarchy level from section path depth
            level = section_title.count(" > ")
            
            # Find parent
            parent_id = None
            if level > 0:
                parent_path = " > ".join(section_title.split(" > ")[:-1])
                # Find parent node by title match (simplified)
                for node in nodes.values():
                    if node.title == parent_path:
                        parent_id = node.node_id
                        break
            
            # Build node
            node = self._build_node(section_title, section_ldus, level, parent_id, node_counter)
            nodes[node.node_id] = node
            
            if level == 0:
                root_nodes.append(node)
        
        # Build parent-child relationships
        for node in nodes.values():
            if node.parent_node_id and node.parent_node_id in nodes:
                parent = nodes[node.parent_node_id]
                if node.node_id not in parent.child_node_ids:
                    parent.child_node_ids.append(node.node_id)
        
        # Calculate statistics
        max_depth = max((n.level for n in nodes.values()), default=0)
        total_pages = max((n.page_end for n in nodes.values()), default=0)
        
        end_time = time.time()
        
        return PageIndex(
            doc_id=doc_id,
            filename=filename,
            root_nodes=root_nodes,
            all_nodes=nodes,
            total_nodes=len(nodes),
            max_depth=max_depth,
            total_pages_covered=total_pages,
            summary_model_used=self.summary_model,
            generation_time_seconds=round(end_time - start_time, 2)
        )
    
    def save(self, pageindex: PageIndex, output_dir: str = ".refinery/pageindex") -> Path:
        """Save PageIndex to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use doc_id for filename
        index_file = output_path / f"{pageindex.doc_id}_pageindex.json"
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(pageindex.model_dump_json(indent=2))
        
        return index_file


def main():
    """CLI entry point for PageIndex building."""
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.indexer <doc_id>")
        sys.exit(1)
    
    doc_id = sys.argv[1]
    
    # Load LDUs (would come from chunking pipeline)
    # For now, this is a placeholder
    print(f"🔧 Building PageIndex for: {doc_id}")
    print("  (Full integration with chunking pipeline in Task 07)")


if __name__ == "__main__":
    main()



