# scripts/fix_query_agent.py
"""Fix Pydantic validation issues in query_agent.py"""

import re
from pathlib import Path

file_path = Path("src/agents/query_agent.py")
content = file_path.read_text(encoding='utf-8')

# Fix 1: Add json import if not present
if 'import json' not in content:
    content = content.replace(
        'from src.models.ldu import LDU',
        'import json\nfrom src.models.ldu import LDU'
    )
    print("✓ Added import json")

# Fix 2: Fix page_number parsing - replace the problematic line
old_page_number = "page_number=fact.get('page_refs', [1])[0] if fact.get('page_refs') else 1,"
new_page_number = '''page_refs_raw = fact.get('page_refs')
                page_refs_parsed = json.loads(page_refs_raw) if isinstance(page_refs_raw, str) else (page_refs_raw if page_refs_raw else [1])
                page_number=page_refs_parsed[0] if page_refs_parsed else 1,'''

content = content.replace(old_page_number, new_page_number)
print("✓ Fixed page_number parsing")

# Fix 3: Fix extraction_strategy literal
content = content.replace('extraction_strategy="structured_query"', 'extraction_strategy="layout_aware"')
print("✓ Fixed extraction_strategy literal")

# Write back
file_path.write_text(content, encoding='utf-8')
print("✓ Saved src/agents/query_agent.py")
