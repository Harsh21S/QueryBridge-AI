# backend/embedding.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(text_chunks):
    return model.encode(text_chunks, convert_to_tensor=True)

def embed_query(query):
    return model.encode(query, convert_to_tensor=True)
