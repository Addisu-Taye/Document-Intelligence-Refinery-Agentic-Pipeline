# Domain Notes: Document Intelligence Refinery

## 1. Empirical Analysis: Corpus Document Profiling

### Methodology
Used `pdfplumber` to analyze character density, image ratio, and table presence across sampled pages (1, 3, 5, 10, last) of each corpus document.

### Key Findings Table

| Document | Expected | Median Char Density | Median Image Ratio | Actual Classification |
|----------|----------|---------------------|---------------------|-----------------------|
| **CBE Annual Report** | Native Digital | 0.00053 (mixed) | 0.666 (mixed) | **MIXED** (image covers + text content) |
| **Audit Report** | Scanned | 0.000076 | 0.672 | **SCANNED** (sparse header text only) |

### 🔴 Critical Insight #1: Cover Page Bias
Analyzing only the first page misclassifies documents. The CBE report's first 2 pages are image-based covers, but page 3+ contain extractable text.
- **Mitigation**: Sample pages across the document (not just first 3).
- **Triage Rule**: Use **median** metrics across sampled pages, not first-page metrics.

### 🔴 Critical Insight #2: Sparse Text Layers in Scanned PDFs
The "scanned" Audit Report has 116 characters on page 1 (likely headers/footers). Relying on `char_count > 0` would misclassify it.
- **Mitigation**: Require **multiple pages** with meaningful text density to classify as "native".
- **Triage Rule**: `if pages_with_text < 2 → scanned_image`

### 🔴 Critical Insight #3: BBox Key Absence
pdfplumber char objects do not guarantee a `bbox` key. Must compute bounding box from `x0, x1, y0, y1`:
```python
bbox = (char['x0'], char['top'], char['x1'], char['bottom'])## 2. Extraction ## 2. 


## 2. Extraction Strategy Decision Tree



```mermaid
graph TD
    A[New Document] --> B[Sample 5 pages: 1,3,5,10,last]
    B --> C[Compute median char_density, image_ratio]
    C --> D{median_density > 0.001 AND median_image_ratio < 0.4?}
    D -->|Yes| E[Strategy A: Fast Text]
    D -->|No| F{Table detected via layout heuristic?}
    F -->|Yes| G[Strategy B: Layout-Aware]
    F -->|No| H[Strategy C: Vision-Augmented]
    E --> I{Confidence > 0.75?}
    I -->|No| G
    G --> J{Confidence > 0.75?}
    J -->|No| H

## 3. Pipeline Diagram
```mermaid
    graph LR
    A[Ingest PDF] --> B[Triage Agent]
    B --> C[Sample Pages + Compute Metrics]
    C --> D{Classification}
    D -->|Native/Simple| E[Strategy A: FastText]
    D -->|Complex/Table| F[Strategy B: Layout]
    D -->|Scanned/LowConf| G[Strategy C: VLM]
    E --> H[Confidence Gate]
    F --> H
    G --> H
    H -->|Low| F
    H -->|Low| G
    H -->|High| I[Semantic Chunking]
    I --> J[PageIndex Builder]
    J --> K[Query Agent]
    I --> L[Extraction Ledger]

    ## 4. Failure Modes & Mitigations

| Failure Mode | Root Cause | Observed In | Mitigation Strategy |
|--------------|------------|-------------|-------------------|
| **Cover page misclassification** | First-page bias: covers are often image-heavy even in native PDFs | CBE Report (pages 1-2) | Sample pages across document (1,3,5,10,last); use **median** metrics, not first-page metrics |
| **Sparse text layer in scanned PDFs** | Headers/footers/metadata may have extractable text even when body is scanned | Audit Report (page 1 header) | Require ≥2 sampled pages with `char_density > 0.0005` to classify as "native" |
| **Missing `bbox` key in pdfplumber chars** | pdfplumber char dict API varies; `bbox` not guaranteed | All documents | Compute bbox from `x0, x1, y0, y1` keys: `bbox = (char['x0'], char['top'], char['x1'], char['bottom'])` |
| **Table structure loss with fast text** | Naive text extraction flattens multi-column tables | CBE Report financial tables | Escalate to Strategy B (Docling/MinerU) when `table_count > 0 AND confidence < 0.75` |
| **Figure-caption separation** | Chunking by token count severs semantic units | All documents | Enforce chunking rule: "figure caption stored as metadata of parent figure chunk" |
| **VLM cost overrun** | Unbounded escalation to expensive vision models | Any document | Implement `budget_guard`: max $2.00/doc, log to `extraction_ledger.jsonl`, halt if exceeded |
| **Cross-reference resolution failure** | "See Table 3" links break when chunks are isolated | FTA Report, Tax Report | Store cross-references as chunk relationships; resolve during query time using PageIndex |
| **Multi-column reading order errors** | Left-to-right text dump ignores column flow | CBE Report, FTA Report | Use layout-aware extraction (Strategy B) for `layout_complexity=multi_column` |
| **Numerical precision loss in OCR** | Scanned tables may misread "1.2B" as "1.28" or "l.2B" | Audit Report, Tax Report | Post-process extracted numbers with validation rules; flag low-confidence numerics for human review |
| **Language detection errors on mixed docs** | Keyword-based classifier fails on code-switching | All Ethiopian docs (Amharic + English) | Use domain_hint as soft signal; fallback to VLM classification if confidence < 0.8 |

### Key Engineering Principles Derived
1. **Never trust a single signal**: Combine char density + image ratio + table count + font metadata for triage.
2. **Escalate early, not late**: If Strategy A confidence < 0.75, retry with Strategy B immediately—don't pass garbage downstream.
3. **Log everything**: Every extraction decision, confidence score, and cost estimate goes to `extraction_ledger.jsonl` for audit.
4. **Budget before beauty**: Vision models are powerful but expensive—always enforce cost caps per document.
5. **Provenance is non-negotiable**: Every extracted fact must carry `(doc_id, page_number, bbox, content_hash)` for verification.