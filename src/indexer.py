"""
Build and persist the vector index (ChromaDB) and BM25 index.

Uses SiliconFlow API for embeddings (BAAI/bge-m3 via cloud).

Usage:
    build_index()              # full build
    build_index(incremental=True)  # only re-index changed/new files
    col, bm25, bm25_docs = load_index()
"""
import os, json, pickle, time, hashlib, glob as _glob
from typing import List, Dict, Any

import chromadb
from rank_bm25 import BM25Okapi
import openai

from src.config import (
    DATA_DIR, VECTOR_DB_DIR, BM25_DIR,
    EMBED_MODEL, SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL
)
from src.chunker import chunk_document

COLLECTION_NAME = "ophthal_kb"
BM25_PICKLE     = os.path.join(BM25_DIR, "bm25_index.pkl")
BM25_DOCS_FILE  = os.path.join(BM25_DIR, "bm25_docs.json")
MANIFEST_FILE   = os.path.join(VECTOR_DB_DIR, "manifest.json")

EMBED_BATCH_SIZE = 64
EMBED_WORKERS    = 4


# ── Embedding ──────────────────────────────────────────────────────────────

def _embed_one_batch(args):
    model, api_key, base_url, texts, batch_idx = args
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    for attempt in range(3):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return batch_idx, [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


class SiliconFlowEmbeddingFunction:
    """ChromaDB-compatible embedding function using SiliconFlow API (concurrent)."""

    def __init__(self, model: str = EMBED_MODEL):
        self.model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        batches = [
            (self.model, SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL,
             input[i:i + EMBED_BATCH_SIZE], i // EMBED_BATCH_SIZE)
            for i in range(0, len(input), EMBED_BATCH_SIZE)
        ]
        results = {}
        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
            futures = {pool.submit(_embed_one_batch, b): b[4] for b in batches}
            for fut in as_completed(futures):
                idx, embeddings = fut.result()
                results[idx] = embeddings
        all_embeddings = []
        for i in sorted(results):
            all_embeddings.extend(results[i])
        return all_embeddings


# ── BM25 tokenizer ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    import re
    tokens = [t for t in re.findall(r'[一-鿿]|[a-zA-Z0-9]+', text.lower())]
    return tokens if tokens else ["<empty>"]


# ── Manifest (incremental tracking) ────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> Dict:
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: Dict):
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _enumerate_db_files(data_dir: str) -> List[Dict]:
    """
    Return list of {db, source, path} for every indexable file.
    Priority: full.md (incl. subdir) > chapters/*.md > single *.md
    """
    entries = []
    db_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("database")
    )
    for db_name in db_dirs:
        db_path = os.path.join(data_dir, db_name)
        full_md = os.path.join(db_path, "full.md")

        # Check one level deep (e.g. BCSC_1/full.md)
        if not os.path.exists(full_md):
            matches = _glob.glob(os.path.join(db_path, "*", "full.md"))
            if matches:
                full_md = matches[0]

        if os.path.exists(full_md) and os.path.getsize(full_md) > 100:
            entries.append({"db": db_name, "source": "full.md", "path": full_md})
            continue

        chapters_dir = os.path.join(db_path, "chapters")
        if os.path.exists(chapters_dir):
            for fname in sorted(f for f in os.listdir(chapters_dir) if f.endswith(".md")):
                entries.append({
                    "db": db_name,
                    "source": fname,
                    "path": os.path.join(chapters_dir, fname),
                })
            continue

        for fname in sorted(f for f in os.listdir(db_path) if f.endswith(".md")):
            entries.append({"db": db_name, "source": fname, "path": os.path.join(db_path, fname)})

    return entries


# ── Upsert helper ──────────────────────────────────────────────────────────

def _upsert_chunks(col, chunks: List[Dict], embed_fn, label: str = ""):
    batch_size = 128
    t0 = time.time()
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        col.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "source": c["source"],
                "db": c["db"],
                "parent_text": c["parent_text"][:2000],
                "title": c.get("title", ""),
            } for c in batch],
        )
        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        print(f"  {label}{done}/{total} chunks | {speed:.1f} chunks/s | ETA {eta/60:.1f} min")


def _rebuild_bm25(bm25_docs: List[Dict]):
    corpus_tokens = [_tokenize(d["text"]) for d in bm25_docs]
    bm25 = BM25Okapi(corpus_tokens)
    with open(BM25_PICKLE, "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(bm25_docs, f, ensure_ascii=False)
    print(f"BM25: {len(bm25_docs)} documents indexed.")


# ── Public API ─────────────────────────────────────────────────────────────

def build_index(data_dir: str = DATA_DIR, force: bool = False, incremental: bool = False):
    """
    Build or update the ChromaDB + BM25 index.

    force=True      → delete everything and rebuild from scratch
    incremental=True → only re-index files whose content has changed (saves API tokens)
    default         → full build only if index doesn't exist yet
    """
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    os.makedirs(BM25_DIR, exist_ok=True)

    embed_fn = SiliconFlowEmbeddingFunction()
    client   = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # ── Full rebuild ────────────────────────────────────────────────────────
    if force:
        print("Force rebuild: deleting existing index...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        col = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        manifest = {}
        bm25_docs_existing = []

    else:
        # Try to get existing collection
        try:
            col = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
            existing_count = col.count()
        except Exception:
            col = client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            existing_count = 0

        if not incremental and existing_count > 0 and os.path.exists(BM25_PICKLE):
            print(f"Index already exists ({existing_count} chunks). "
                  f"Use force=True to rebuild or incremental=True to update.")
            return

        manifest = _load_manifest()
        # Load existing BM25 docs
        if os.path.exists(BM25_DOCS_FILE):
            with open(BM25_DOCS_FILE, encoding="utf-8") as f:
                bm25_docs_existing = json.load(f)
        else:
            bm25_docs_existing = []

    # ── Enumerate all files ─────────────────────────────────────────────────
    db_files = _enumerate_db_files(data_dir)

    # If incremental but no manifest yet, bootstrap it from current files
    # without re-embedding — assume the existing index is already up to date
    if incremental and not force and not manifest and col.count() > 0:
        print("No manifest found but index exists. Bootstrapping manifest from current files...")
        for entry in db_files:
            key = f"{entry['db']}::{entry['source']}"
            # Count chunks already in collection for this file
            try:
                existing = col.get(where={"$and": [
                    {"db":     {"$eq": entry["db"]}},
                    {"source": {"$eq": entry["source"]}},
                ]}, include=[])
                chunk_count = len(existing["ids"])
            except Exception:
                chunk_count = 0
            manifest[key] = {
                "hash":        _file_hash(entry["path"]),
                "path":        entry["path"],
                "chunk_count": chunk_count,
                "indexed_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            print(f"  manifest: {key} ({chunk_count} chunks)")
        _save_manifest(manifest)
        print(f"Manifest bootstrapped ({len(manifest)} files). Run incremental again to detect changes.")
        return
    print(f"\nFound {len(db_files)} indexable files.")

    new_bm25_docs = list(bm25_docs_existing)  # start from existing
    changed = 0
    skipped = 0
    t_total = time.time()

    for entry in db_files:
        db_name = entry["db"]
        source  = entry["source"]
        path    = entry["path"]
        key     = f"{db_name}::{source}"

        current_hash = _file_hash(path)
        manifest_entry = manifest.get(key, {})

        if not force and manifest_entry.get("hash") == current_hash:
            print(f"  [SKIP] {key} (unchanged)")
            skipped += 1
            continue

        # File is new or changed — re-index it
        action = "UPDATE" if key in manifest else "ADD"
        print(f"\n  [{action}] {key}")

        # Remove old chunks from ChromaDB
        if action == "UPDATE":
            try:
                col.delete(where={"$and": [
                    {"db":     {"$eq": db_name}},
                    {"source": {"$eq": source}},
                ]})
                print(f"    Deleted old chunks for {key}")
            except Exception as e:
                print(f"    Warning: could not delete old chunks: {e}")

            # Remove from bm25_docs_existing
            new_bm25_docs = [d for d in new_bm25_docs
                             if not (d["db"] == db_name and d["source"] == source)]

        # Load and chunk the file
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if len(text.strip()) < 30:
            print(f"    Skipping empty file.")
            continue

        chunks = chunk_document(text, source=source, db_name=db_name)
        print(f"    {len(chunks)} child chunks generated")

        # Upsert into ChromaDB
        _upsert_chunks(col, chunks, embed_fn, label=f"    ChromaDB {key}: ")

        # Extend BM25 docs
        new_bm25_docs.extend([{
            "id": c["id"], "text": c["text"],
            "parent_text": c["parent_text"],
            "source": c["source"], "db": c["db"],
        } for c in chunks])

        # Update manifest
        manifest[key] = {
            "hash":        current_hash,
            "path":        path,
            "chunk_count": len(chunks),
            "indexed_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _save_manifest(manifest)
        changed += 1

    # ── Rebuild BM25 (always, even if only one file changed) ────────────────
    if changed > 0 or force:
        print(f"\nRebuilding BM25 ({len(new_bm25_docs)} docs)...")
        _rebuild_bm25(new_bm25_docs)
    else:
        print("\nNo files changed — BM25 unchanged.")

    elapsed = time.time() - t_total
    print(f"\nDone! changed={changed} skipped={skipped} "
          f"total_chunks={col.count()} time={elapsed/60:.1f} min")


def load_index():
    """Load ChromaDB collection and BM25 index. Returns (collection, bm25, bm25_docs)."""
    embed_fn = SiliconFlowEmbeddingFunction()
    client   = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    col      = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    with open(BM25_PICKLE, "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_DOCS_FILE, encoding="utf-8") as f:
        bm25_docs = json.load(f)

    return col, bm25, bm25_docs
