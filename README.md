# DocMind RAG — Backend

FastAPI backend for multi-user RAG (Retrieval-Augmented Generation) with Pinecone vector search and Gemini LLM.

## Tech Stack

- **FastAPI** — REST API framework
- **Pinecone** — Vector database (user-isolated search)
- **Sentence Transformers** — `all-MiniLM-L6-v2` local embeddings (no API cost)
- **Gemini** — LLM for answer generation (via OpenAI-compatible endpoint)
- **SQLite** — User and document metadata storage
- **bcrypt + JWT** — Auth

## Project Structure
```
backend/
├── app/
│   ├── main.py               # FastAPI app entry point
│   ├── database.py           # SQLite setup
│   ├── models.py             # JWT, bcrypt, user models
│   ├── dependencies.py       # Auth middleware
│   ├── pinecone_service.py   # Chunking, embedding, Pinecone upsert/query/delete
│   ├── text_extraction.py    # PDF, DOCX, TXT text extraction
│   └── routers/
│       ├── auth.py           # /auth/signup, /auth/login
│       ├── documents.py      # /documents/upload, list, delete
│       └── chat.py           # /chat/ask (RAG pipeline)
├── create_index.py           # Run once to create Pinecone index
├── render.yaml               # Render deployment config
├── requirements.txt
└── .env.example
```

## Local Setup

### 1. Clone and navigate
```bash
git clone https://github.com/your-username/your-repo.git
cd backend
```
clone with exact url from github

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install "numpy<2"
pip install -r requirements.txt
```

### 4. Configure environment
```bash
copy .env.example .env      # Windows
cp .env.example .env        # Mac/Linux
```

Fill in your `.env`:
```env
OPEN_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=pcsk_your_key_here
PINECONE_INDEX_NAME=rag-index
SECRET_KEY=your_long_random_secret
DB_PATH=rag_app.db
```

### 5. Create Pinecone index (run once)
```bash
python create_index.py
```

> Make sure your Pinecone index dimension is set to **384** (all-MiniLM-L6-v2 output).

### 6. Start server
```bash
uvicorn app.main:app --reload
```

API runs at `http://localhost:8000`  
Swagger docs at `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/signup` | ❌ | Create account |
| POST | `/auth/login` | ❌ | Get JWT token |
| POST | `/documents/upload` | ✅ | Upload + index document |
| GET | `/documents/list` | ✅ | List your documents |
| DELETE | `/documents/{id}` | ✅ | Delete doc + Pinecone vectors |
| POST | `/chat/ask` | ✅ | Ask question (RAG) |

---

## Supported File Types

| Type | Extension | Max Size |
|------|-----------|----------|
| PDF | `.pdf` | 20 MB |
| Word | `.docx` | 20 MB |
| Plain text | `.txt` | 20 MB |
| Markdown | `.md` | 20 MB |

---

## User Isolation

Every Pinecone query includes a hard metadata filter:
```python
filter={"user_id": {"$eq": user_id}}
```
It is **impossible** for one user to retrieve another user's document chunks.

---

## Deploy to Render (Free)

1. Push this folder to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo, set root directory to `backend/`
4. Render will auto-detect `render.yaml`
5. Add your secret env vars in the Render dashboard:
   - `OPEN_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME`

> ⚠️ Render free tier sleeps after 15 min inactivity (~30s cold start).  
> ⚠️ `/tmp/rag_app.db` resets on redeploy. Use Render's persistent disk for production.
