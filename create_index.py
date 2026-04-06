"""
Run this ONCE to create your Pinecone index.
  python create_index.py
"""
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

#You can create a Index in Pinecone UI
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,          # text-embedding-3-small output dim
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"   # free-tier region
        )
    )
    print(f"✓ Index '{INDEX_NAME}' created")
else:
    #if index exists already it won't create again
    print(f"Index '{INDEX_NAME}' already exists")

info = pc.describe_index(INDEX_NAME)
print(f"Index status: {info.status}")
