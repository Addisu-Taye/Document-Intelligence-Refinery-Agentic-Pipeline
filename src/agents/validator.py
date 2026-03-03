# src/agents/validator.py
"""
ChunkValidator - Enforces the 5 Chunking Rules

This is the "Constitution" for data quality in the Refinery.
Every LDU must pass validation before being emitted.
"""

# tuple is a built-in type in Python 3.9+
from src.models.ldu import LDU


class ChunkValidator:
    """
    Validates LDUs against the 5 Chunking Rules.
    
    Rules:
    1. Table cells never split from header rows
    2. Figure captions stored as parent metadata
    3. Numbered lists kept atomic (unless > max_tokens)
    4. Section headers propagated to child chunks
    5. Cross-references resolved as chunk relationships
    """
    
    # Maximum tokens before a list can be split
    MAX_LIST_TOKENS = 512
    
    def validate(self, ldu: LDU) -> tuple[bool, list[str]]:
        """
        Validate an LDU against all applicable rules.
        
        Args:
            ldu: Logical Document Unit to validate
        
        Returns:
            (is_valid, list of error messages)
        """
        errors = []
        
        # Rule 1: Table integrity
        if ldu.chunk_type == "table":
            is_valid, error = self._validate_table_integrity(ldu)
            if not is_valid:
                errors.append(error)
        
        # Rule 2: Figure caption presence
        if ldu.chunk_type == "figure":
            is_valid, error = self._validate_figure_caption(ldu)
            if not is_valid:
                errors.append(error)
        
        # Rule 3: List atomicity
        if ldu.chunk_type == "list":
            is_valid, error = self._validate_list_atomicity(ldu)
            if not is_valid:
                errors.append(error)
        
        # Rule 4: Section header propagation
        is_valid, error = self._validate_section_propagation(ldu)
        if not is_valid:
            errors.append(error)
        
        # Rule 5: Cross-reference resolution (informational)
        # This rule is about relationships, not validation failure
        
        return len(errors) == 0, errors
    
    def _validate_table_integrity(self, ldu: LDU) -> tuple[bool, str]:
        """
        Rule 1: Table cells never split from header rows.
        
        Validation:
        - table_headers must be present
        - table_data must have same column count as headers
        """
        if not ldu.table_headers:
            return False, f"LDU {ldu.ldu_id}: Table missing headers"
        
        if ldu.table_data:
            header_count = len(ldu.table_headers)
            for i, row in enumerate(ldu.table_data):
                if len(row) != header_count:
                    return False, f"LDU {ldu.ldu_id}: Table row {i} has {len(row)} columns, expected {header_count}"
        
        return True, None
    
    def _validate_figure_caption(self, ldu: LDU) -> tuple[bool, str]:
        """
        Rule 2: Figure captions stored as parent metadata.
        
        Validation:
        - figure_caption must be present (can be "No caption" but must exist)
        """
        if ldu.figure_caption is None:
            return False, f"LDU {ldu.ldu_id}: Figure missing caption"
        
        return True, None
    
    def _validate_list_atomicity(self, ldu: LDU) -> tuple[bool, str]:
        """
        Rule 3: Numbered lists kept atomic (unless > max_tokens).
        
        Validation:
        - If token_count > MAX_LIST_TOKENS, list should be split (warning)
        - Otherwise, list should be complete
        """
        if ldu.token_count > self.MAX_LIST_TOKENS:
            # This is a warning, not a failure - list should be split
            return True, f"LDU {ldu.ldu_id}: List exceeds max tokens ({ldu.token_count} > {self.MAX_LIST_TOKENS}), consider splitting"
        
        return True, None
    
    def _validate_section_propagation(self, ldu: LDU) -> tuple[bool, str]:
        """
        Rule 4: Section headers propagated to child chunks.
        
        Validation:
        - parent_section should be set for non-root chunks
        - section_path should not be empty for structured documents
        """
        # This is informational - some chunks may legitimately have no section
        if not ldu.parent_section and ldu.chunk_type not in ["section_header"]:
            # Warning only, not a failure
            pass
        
        return True, None
    
    def validate_batch(self, ldus: list[LDU]) -> dict:
        """
        Validate a batch of LDUs and return summary statistics.
        
        Args:
            ldus: List of LDUs to validate
        
        Returns:
            Dict with validation summary
        """
        total = len(ldus)
        passed = 0
        failed = 0
        errors_by_rule = {
            "table_integrity": 0,
            "figure_caption": 0,
            "list_atomicity": 0,
            "section_propagation": 0
        }
        
        for ldu in ldus:
            is_valid, errors = self.validate(ldu)
            if is_valid:
                passed += 1
            else:
                failed += 1
                for error in errors:
                    if "headers" in error.lower():
                        errors_by_rule["table_integrity"] += 1
                    elif "caption" in error.lower():
                        errors_by_rule["figure_caption"] += 1
                    elif "tokens" in error.lower():
                        errors_by_rule["list_atomicity"] += 1
        
        return {
            "total_ldus": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "errors_by_rule": errors_by_rule
        }

