from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from app.dependencies import get_current_user
from app.pinecone_service import query_pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

router = APIRouter()

# Gemini via Google's OpenAI-compatible endpoint
client = OpenAI(
    api_key=os.getenv("OPEN_API_KEY"),
    base_url="https://api.together.xyz/v1"
)

class ChatRequest(BaseModel):
    question: str
    doc_id:   Optional[str] = None  # Optional: restrict to a specific doc
    top_k:    int = 5

class Source(BaseModel):
    filename: str
    doc_id:   str
    chunk_index: int
    score:    float
    snippet:  str

class ChatResponse(BaseModel):
    answer:  str
    sources: List[Source]

SYSTEM_PROMPT = """
You are a precise and reliable document assistant designed to answer questions using ONLY the provided context.

Guidelines:
1. Strict Source Limitation
   - Answer ONLY using the provided context chunks.
   - Do NOT use external knowledge, assumptions, or prior training information.

2. Missing Information Handling
   - If the answer cannot be found in the context, respond with:
     "I couldn't find this information in your documents."

3. Citations
   - Always cite the document name or metadata source for every answer.
   - If multiple documents are used, list all of them.

4. Response Style
   - Be concise, clear, and factual.
   - Use bullet points for lists.
   - Avoid unnecessary explanations.

5. Conflicting Information
   - If the context contains conflicting information:
     • Mention the conflict
     • Cite the sources for each conflicting statement
     • Do NOT decide which one is correct unless the context clearly resolves it.

6. Partial Answers
   - If only part of the question can be answered from the context, answer that part and state what information is missing.

7. Quoting
   - Prefer paraphrasing but include short quotes when helpful.
   - Do not fabricate text that does not appear in the context.
"""

def build_context(chunks: List[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | File: {c['filename']} | Score: {c['score']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)

@router.post("/ask", response_model=ChatResponse)
def ask_question(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    
    # Retrieve relevant chunks (user-isolated via Pinecone filter)
    chunks = query_pinecone(req.question, user_id, top_k=req.top_k)
    
    if not chunks:
        return ChatResponse(
            answer="I couldn't find any relevant information in your documents. Please upload documents first.",
            sources=[]
        )
    
    context = build_context(chunks)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"}
    ]
    
    try:
        resp = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # swap to gemini-1.5-pro for higher quality
            messages=messages,
            temperature=0.1,           # Low temp for factual accuracy
            max_tokens=1024,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(500, f"LLM error: {str(e)}")
    
    sources = [
        Source(
            filename=c["filename"],
            doc_id=c["doc_id"],
            chunk_index=c["chunk_index"],
            score=c["score"],
            snippet=c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"]
        )
        for c in chunks
    ]
    
    return ChatResponse(answer=answer, sources=sources)