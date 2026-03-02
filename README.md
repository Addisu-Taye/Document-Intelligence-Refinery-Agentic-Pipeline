# Document Intelligence Refinery

> TRP1 Week 3 Challenge: Engineering Agentic Pipelines for Unstructured Document Extraction at Enterprise Scale  
> **Author**: Addisu Taye <addtaye@gmail.com>  
> **Status**: ✅ Task 01 Complete — Domain Onboarding & Project Initialization  

---

## 🎯 Business Objective

Enterprises lose billions annually because institutional knowledge is trapped in PDFs, scanned reports, and slide decks. This project builds **The Document Intelligence Refinery**—a production-grade, multi-stage agentic pipeline that ingests heterogeneous documents and emits structured, queryable, spatially-indexed knowledge with full provenance tracking.

---

## ▶️ Run Corpus Analysis (Task 01)

```bash
# Analyze character density, image ratio, and tables across corpus
poetry run python scripts/analyze_corpus.py
```

---

## ▶️ Run Full Pipeline (Task 02+)

```bash
# Process a single document end-to-end
poetry run python src/main.py --doc corpus/"CBE ANNUAL REPORT 2023-24.pdf"

# Process all corpus documents
poetry run python src/main.py --all
```

---

## 🧪 Run Tests

```bash
pytest
```

---

## 📁 Project Structure

```
document-refinery/
├── .git/                          # Version control
├── .gitignore                     # Ignored files
├── .refinery/                     # Runtime artifacts (tracked per challenge spec)
│   ├── profiles/                  # DocumentProfile JSONs (one per corpus doc)
│   ├── extraction_ledger.jsonl    # Audit log of all extraction decisions
│   └── pageindex/                 # PageIndex trees (final deliverable)
├── corpus/                        # Input documents (4 PDFs)
├── scripts/
│   └── analyze_corpus.py
├── src/
│   ├── main.py
│   ├── agents/
│   │   ├── triage.py
│   │   └── extractor.py
│   └── models/
├── rubric/
│   └── extraction_rules.yaml
└── README.md
```

---

## 📚 Corpus Documents

| Class | Document                    | Type                 | Key Challenge |
|-------|-----------------------------|----------------------|--------------|
| A     | CBE Annual Report 2023-24   | Native Digital (Mixed) | Multi-column layouts, embedded financial tables, footnotes |
| B     | Audit Report 2023           | Scanned Image        | No character stream—pure image; requires OCR/VLM |
| C     | FTA Performance Survey      | Mixed Layout         | Hierarchical sections, narrative + tables + findings |
| D     | Tax Expenditure Report      | Table-Heavy          | Multi-year fiscal data, numerical precision, category hierarchies |

---

## 🔑 Key Features

### 1️⃣ Triage Agent

Classifies documents by:

- **Origin Type**: `native_digital | scanned_image | mixed | form_fillable`  
- **Layout Complexity**: `single_column | multi_column | table_heavy | figure_heavy`  
- **Domain Hint**: `financial | legal | technical | medical | general`  
- **Estimated Cost**: `fast_text_sufficient | needs_layout_model | needs_vision_model`  

---

### 2️⃣ Multi-Strategy Extraction with Escalation Guard

| Strategy | Tool                          | Cost        | Trigger Condition |
|----------|-------------------------------|------------|------------------|
| A: Fast Text | pdfplumber / pymupdf       | $0.00      | Native + simple layout + high confidence |
| B: Layout-Aware | Docling / MinerU       | $0.00 (local) | Multi-column OR tables OR Strategy A confidence < 0.75 |
| C: Vision-Augmented | GPT-4o-mini via OpenRouter | ~$0.02/page | Scanned OR handwriting OR Strategy B confidence < 0.80 |

**Escalation Guard:**  
Low-confidence extractions automatically retry with the next strategy—preventing "garbage in, hallucination out" failures.

---

### 3️⃣ Semantic Chunking with 5 Constitutional Rules

1. Table cells never split from header rows  
2. Figure captions stored as parent metadata  
3. Numbered lists kept atomic (unless `> max_tokens`)  
4. Section headers propagated to child chunks  
5. Cross-references ("see Table 3") resolved as chunk relationships  

---

### 4️⃣ PageIndex Navigation Tree

Hierarchical "smart table of contents" enabling LLMs to:

- Navigate to relevant sections without full-document embedding search  
- Retrieve context-aware chunks with parent section metadata  
- Generate section summaries for query routing  

---

### 5️⃣ Full Provenance Tracking

Every answer includes a `ProvenanceChain`:

```json
{
  "document_id": "doc_001",
  "page_number": 14,
  "chunk_id": "chunk_14_03",
  "extraction_strategy": "layout_aware",
  "confidence_score": 0.87,
  "timestamp": "2026-03-02T12:45:11Z"
}
```

---

## ⚙️ Configuration

Edit `rubric/extraction_rules.yaml` to adjust:

```yaml
confidence_thresholds:
  strategy_a: 0.75
  strategy_b: 0.80

max_tokens_per_chunk: 800
enable_cross_reference_resolution: true
enable_table_atomicity: true
```

---

## 📦 Deployment

### Local Development

```bash
poetry install
poetry run python src/main.py --all
```

### Docker (Recommended for Production)

```bash
docker build -t document-refinery .
docker run -v $(pwd)/corpus:/app/corpus document-refinery
```

---

## 📄 Deliverables Timeline

### Interim (Thursday 03:00 UTC)

- ✅ `DOMAIN_NOTES.md` with decision tree, pipeline diagram, failure modes  
- ✅ `pyproject.toml` with locked dependencies  
- ✅ `README.md` with setup instructions  
- ✅ Core models (`src/models/`) and agents (`triage.py`, `extractor.py`)  
- ✅ `extraction_rules.yaml` with thresholds  
- ✅ `.refinery/profiles/` and `extraction_ledger.jsonl` for all 4 corpus docs  

---

### Final (Sunday 03:00 UTC)

- Chunking engine  
- PageIndex builder  
- Query agent  
- FactTable extractor + SQLite backend  
- Vector store ingestion (ChromaDB)  
- Audit mode with claim verification  
- Dockerfile + deployment docs  
- 5-minute video demo following protocol  

---

## 🔗 References

- MinerU Architecture  
- Docling Document Model  
- PageIndex Navigation  
- Chunkr RAG Optimization  
- Marker PDF-to-Markdown  

---

## 📝 License

MIT License — See `LICENSE` file for details.

---

## 💡 FDE Insight

The ability to onboard to a new document domain in 24 hours—understanding its structure, failure modes, and correct extraction strategy—is precisely what separates a forward-deployed engineer from a developer who can only work in familiar territory.

---

