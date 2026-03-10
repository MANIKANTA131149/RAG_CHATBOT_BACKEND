import os
import re
import json
import uuid
from typing import List, Dict, Any
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# ── Clients ──────────────────────────────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "rag-index"))

# all-MiniLM-L6-v2: fast, lightweight, 384-dim, fully free/offline
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM  = 384

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # tokens ≈ ~400 words – good recall/precision balance
CHUNK_OVERLAP = 80    # ~15 % overlap prevents context loss at boundaries

def _approx_token_count(text: str) -> int:
    """Fast approximation: 1 token ≈ 4 chars."""
    return len(text) // 4

def _split_into_sentences(text: str) -> List[str]:
    """Split on sentence boundaries while keeping the delimiter."""
    pattern = r'(?<=[.!?])\s+'
    parts = re.split(pattern, text.strip())
    return [p.strip() for p in parts if p.strip()]

def semantic_chunk(text: str, chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Sentence-aware sliding-window chunker.

    Strategy:
      1. Split text into sentences.
      2. Greedily pack sentences until chunk_size is reached.
      3. Slide backward by `overlap` tokens before starting the next chunk.
      4. Record char offsets + chunk index as metadata.
    """
    sentences = _split_into_sentences(text)
    chunks: List[Dict[str, Any]] = []
    current_tokens: List[str] = []  # sentences in current window
    current_size = 0
    char_offset = 0

    for sent in sentences:
        sent_tokens = _approx_token_count(sent)

        # If adding this sentence exceeds limit → flush
        if current_size + sent_tokens > chunk_size and current_tokens:
            chunk_text = " ".join(current_tokens)
            chunks.append({
                "text":        chunk_text,
                "chunk_index": len(chunks),
                "char_start":  char_offset,
                "char_end":    char_offset + len(chunk_text),
                "token_count": current_size,
            })

            # --- overlap: retain trailing sentences up to `overlap` tokens ---
            overlap_sentences: List[str] = []
            overlap_size = 0
            for prev_sent in reversed(current_tokens):
                prev_tokens = _approx_token_count(prev_sent)
                if overlap_size + prev_tokens > overlap:
                    break
                overlap_sentences.insert(0, prev_sent)
                overlap_size += prev_tokens

            char_offset += len(chunk_text) + 1
            current_tokens = overlap_sentences
            current_size   = overlap_size

        current_tokens.append(sent)
        current_size += sent_tokens

    # Flush remainder
    if current_tokens:
        chunk_text = " ".join(current_tokens)
        chunks.append({
            "text":        chunk_text,
            "chunk_index": len(chunks),
            "char_start":  char_offset,
            "char_end":    char_offset + len(chunk_text),
            "token_count": current_size,
        })

    return chunks


# ── Embeddings ────────────────────────────────────────────────────────────────
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed with all-MiniLM-L6-v2 — runs locally, no API key needed."""
    vectors = _embedder.encode(texts, batch_size=64, show_progress_bar=False)
    return vectors.tolist()


# ── Pinecone helpers ──────────────────────────────────────────────────────────
def upsert_chunks(chunks: List[Dict], user_id: str,
                  doc_id: str, filename: str) -> List[str]:
    """
    Upsert chunk vectors into Pinecone with rich metadata.

    Metadata fields:
      user_id       – partition key  (NEVER cross-user leakage)
      doc_id        – document identifier
      filename      – human-readable name
      chunk_index   – position in document
      char_start/end – byte offsets for source highlight
      token_count   – chunk length
      text          – stored verbatim for retrieval (no re-fetch needed)
      source_page   – page hint for PDF-derived text (future)
    """
    vectors = embed_texts([c["text"] for c in chunks])
    pinecone_ids = []
    records = []

    for chunk, vec in zip(chunks, vectors):
        pid = f"{user_id}_{doc_id}_{chunk['chunk_index']}_{uuid.uuid4().hex[:8]}"
        records.append({
            "id":     pid,
            "values": vec,
            "metadata": {
                "user_id":     user_id,
                "doc_id":      doc_id,
                "filename":    filename,
                "chunk_index": chunk["chunk_index"],
                "char_start":  chunk["char_start"],
                "char_end":    chunk["char_end"],
                "token_count": chunk["token_count"],
                "text":        chunk["text"][:2000],  # Pinecone 40 KB metadata cap
            }
        })
        pinecone_ids.append(pid)

    # Pinecone upsert in batches of 100
    BATCH = 100
    for i in range(0, len(records), BATCH):
        index.upsert(vectors=records[i:i+BATCH])

    return pinecone_ids


def query_pinecone(query: str, user_id: str, top_k: int = 5) -> List[Dict]:
    """
    Semantic search filtered strictly to `user_id` namespace.
    Uses Pinecone metadata filter to guarantee user isolation.
    """
    q_vec = embed_texts([query])[0]
    result = index.query(
        vector=q_vec,
        top_k=top_k,
        filter={"user_id": {"$eq": user_id}},  # ← hard isolation
        include_metadata=True
    )
    return [
        {
            "text":     m.metadata.get("text", ""),
            "filename": m.metadata.get("filename", ""),
            "doc_id":   m.metadata.get("doc_id", ""),
            "chunk_index": m.metadata.get("chunk_index", 0),
            "score":    round(m.score, 4),
        }
        for m in result.matches
    ]


def delete_doc_vectors(pinecone_ids: List[str]):
    """Delete all vectors belonging to a document."""
    BATCH = 100
    for i in range(0, len(pinecone_ids), BATCH):
        index.delete(ids=pinecone_ids[i:i+BATCH])