# backend/search.py
from sentence_transformers import util

def search(query_embedding, doc_embeddings, top_k=5):
    hits = util.semantic_search(query_embedding, doc_embeddings, top_k=top_k)[0]
    return hits
