from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.dependencies import get_current_user
from app.database import get_db
from app.models import generate_id
from app.pinecone_service import semantic_chunk, upsert_chunks, delete_doc_vectors
from app.text_extraction import extract_text, clean_text
import json
from datetime import datetime

router = APIRouter()

ALLOWED_TYPES = {"pdf", "txt", "docx", "md"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    
    # Validate extension
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"File type .{ext} not supported. Allowed: {ALLOWED_TYPES}")
    
    # Read file
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max size: 20 MB")
    
    # Extract + clean text
    try:
        raw_text = extract_text(file_bytes, file.filename)
        text = clean_text(raw_text)
    except Exception as e:
        raise HTTPException(400, f"Text extraction failed: {str(e)}")
    
    if len(text.strip()) < 50:
        raise HTTPException(400, "File appears empty or unreadable")
    
    # Create DB record
    doc_id = generate_id()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO documents (id, user_id, filename, original_filename, file_size, status)
           VALUES (?, ?, ?, ?, ?, 'processing')""",
        (doc_id, user_id, f"{doc_id}.{ext}", file.filename, len(file_bytes))
    )
    conn.commit()
    conn.close()
    
    # Chunk the text
    chunks = semantic_chunk(text)
    
    # Embed + upsert to Pinecone
    try:
        pinecone_ids = upsert_chunks(chunks, user_id, doc_id, file.filename)
    except Exception as e:
        # Mark as failed
        conn = get_db()
        conn.execute("UPDATE documents SET status='failed' WHERE id=?", (doc_id,))
        conn.commit()
        conn.close()
        raise HTTPException(500, f"Embedding failed: {str(e)}")
    
    # Update DB with success
    conn = get_db()
    conn.execute(
        "UPDATE documents SET status='ready', chunk_count=?, pinecone_ids=? WHERE id=?",
        (len(chunks), json.dumps(pinecone_ids), doc_id)
    )
    conn.commit()
    conn.close()
    
    return {
        "doc_id":      doc_id,
        "filename":    file.filename,
        "chunk_count": len(chunks),
        "status":      "ready",
        "message":     f"Successfully indexed {len(chunks)} chunks"
    }


@router.get("/list")
def list_documents(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, original_filename, chunk_count, file_size, status, created_at FROM documents WHERE user_id=? ORDER BY created_at DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.delete("/{doc_id}")
def delete_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM documents WHERE id=? AND user_id=?", (doc_id, user_id))
    doc = cursor.fetchone()
    
    if not doc:
        conn.close()
        raise HTTPException(404, "Document not found")
    
    # Delete from Pinecone
    pinecone_ids = json.loads(doc["pinecone_ids"] or "[]")
    if pinecone_ids:
        try:
            delete_doc_vectors(pinecone_ids)
        except Exception as e:
            conn.close()
            raise HTTPException(500, f"Pinecone delete failed: {e}")
    
    # Delete from DB
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()
    
    return {"message": "Document deleted successfully", "doc_id": doc_id}
