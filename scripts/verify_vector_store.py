# scripts/verify_vector_store.py
"""Verify ChromaDB 0.5.x vector store is working correctly."""

import sys
import chromadb
from pathlib import Path

print("=" * 60, flush=True)
print(f"VECTOR STORE VERIFICATION (ChromaDB {chromadb.__version__})", flush=True)
print("=" * 60, flush=True)

persist_dir = Path('.refinery/vector_store').resolve()
print(f'\nVector store path: {persist_dir}', flush=True)
print(f'Path exists: {persist_dir.exists()}', flush=True)

if not persist_dir.exists():
    print('ERROR: Vector store directory does not exist!', flush=True)
    sys.exit(1)

print('\nInitializing ChromaDB PersistentClient...', flush=True)
client = chromadb.PersistentClient(
    path=str(persist_dir),
    settings=chromadb.config.Settings(anonymized_telemetry=False)
)
print('Client initialized', flush=True)

collections = client.list_collections()
print(f'\nCollections found: {len(collections)}', flush=True)

if not collections:
    print('ERROR: No collections found!', flush=True)
    print('\nTroubleshooting:', flush=True)
    print('1. Check if chroma.sqlite3 exists in vector_store folder', flush=True)
    print('2. Re-run: poetry run python src/agents/embedder.py', flush=True)
    sys.exit(1)

for col in collections:
    print(f'  - {col.name}', flush=True)
    count = col.count()
    print(f'    Total embeddings: {count}', flush=True)

# Test search
print('\nTesting semantic search...', flush=True)
collection = client.get_collection('ldu_embeddings')

test_queries = ['net profit', 'revenue', 'financial']

for query in test_queries:
    print(f'\n  Query: "{query}"', flush=True)
    results = collection.query(
        query_texts=[query],
        n_results=2,
        include=['documents', 'metadatas', 'distances']
    )
    
    if results['documents'] and results['documents'][0]:
        print(f'    Results: {len(results["documents"][0])}', flush=True)
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f'    {i}. Distance: {dist:.4f}', flush=True)
            print(f'       Page: {meta.get("page_refs", "N/A")}', flush=True)
            print(f'       Content: {doc[:100]}...', flush=True)
    else:
        print('    No results', flush=True)

print('\n' + '=' * 60, flush=True)
print('VERIFICATION COMPLETE', flush=True)
print('=' * 60, flush=True)
