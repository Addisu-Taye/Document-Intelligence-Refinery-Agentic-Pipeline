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

    ## 🗂️ Final Project Structure (Visual Check)