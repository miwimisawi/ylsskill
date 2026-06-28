"""
Parent-Child chunking strategy.

Parent chunks (~1024 chars) preserve context.
Child chunks (~256 chars) are stored in the vector DB for precise retrieval.
Each child carries a reference to its parent's full text.
"""
import os, re, json
from typing import List, Dict, Any
from src.config import PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE, CHUNK_OVERLAP


def _split_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks by character count."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        # Try to break at sentence boundary
        if end < len(text):
            for sep in ("。", ".\n", "!\n", "？", ".", "! ", " "):
                pos = text.rfind(sep, start + size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return [c for c in chunks if len(c) > 20]


def chunk_document(text: str, source: str, db_name: str, metadata: Dict = None) -> List[Dict[str, Any]]:
    """
    Produce parent-child chunk pairs from a document.

    Returns list of child chunk dicts, each containing:
      - id: unique id
      - text: child chunk text (for embedding)
      - parent_text: parent chunk text (for LLM context)
      - source: file name
      - db: database name
      - metadata: extra metadata
    """
    metadata = metadata or {}
    parents = _split_text(text, PARENT_CHUNK_SIZE, CHUNK_OVERLAP // 2)
    records = []
    for p_idx, parent_text in enumerate(parents):
        children = _split_text(parent_text, CHILD_CHUNK_SIZE, CHUNK_OVERLAP)
        for c_idx, child_text in enumerate(children):
            records.append({
                "id": f"{db_name}::{source}::p{p_idx}::c{c_idx}",
                "text": child_text,
                "parent_text": parent_text,
                "source": source,
                "db": db_name,
                "parent_idx": p_idx,
                "child_idx": c_idx,
                **metadata,
            })
    return records


def load_database_chunks(data_dir: str) -> List[Dict[str, Any]]:
    """Load and chunk all databases, return all child chunk records."""
    all_chunks = []
    db_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("database")
    ])

    for db_name in db_dirs:
        db_path = os.path.join(data_dir, db_name)
        full_md = os.path.join(db_path, "full.md")
        chapters_dir = os.path.join(db_path, "chapters")
        toc_path = os.path.join(db_path, "toc.json")

        toc = {}
        if os.path.exists(toc_path):
            with open(toc_path, encoding="utf-8") as f:
                toc = json.load(f)

        # For db1/db2: full.md may be inside a subdirectory (e.g. BCSC_1/)
        if not os.path.exists(full_md):
            import glob as _glob
            matches = _glob.glob(os.path.join(db_path, "*", "full.md"))
            if matches:
                full_md = matches[0]

        if os.path.exists(full_md):
            with open(full_md, encoding="utf-8") as f:
                text = f.read()
            if len(text.strip()) > 100:
                chunks = chunk_document(
                    text, source="full.md", db_name=db_name,
                    metadata={"title": toc.get("title", db_name)}
                )
                all_chunks.extend(chunks)
                print(f"  {db_name}/full.md → {len(chunks)} child chunks")
                continue  # skip chapters if full.md exists with content

        # For db3-db6: use individual chapter files
        if os.path.exists(chapters_dir):
            chapter_files = sorted(f for f in os.listdir(chapters_dir) if f.endswith(".md"))
            db_chunks = []
            for fname in chapter_files:
                fpath = os.path.join(chapters_dir, fname)
                with open(fpath, encoding="utf-8") as f:
                    text = f.read()
                if len(text.strip()) < 30:
                    continue
                chunks = chunk_document(
                    text, source=fname, db_name=db_name,
                    metadata={"title": toc.get("title", db_name)}
                )
                db_chunks.extend(chunks)
            all_chunks.extend(db_chunks)
            print(f"  {db_name}/chapters/ ({len(chapter_files)} files) → {len(db_chunks)} child chunks")
            continue

        # db6: single md file
        md_files = [f for f in os.listdir(db_path) if f.endswith(".md")]
        for fname in md_files:
            with open(os.path.join(db_path, fname), encoding="utf-8") as f:
                text = f.read()
            chunks = chunk_document(
                text, source=fname, db_name=db_name,
                metadata={"title": toc.get("title", db_name)}
            )
            all_chunks.extend(chunks)
            print(f"  {db_name}/{fname} → {len(chunks)} child chunks")

    return all_chunks
