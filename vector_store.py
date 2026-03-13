import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a local lightweight embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

embedding_dim = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(embedding_dim)  # Inner Product (Cosine sim if normalized)

# Maps faiss indices to chunk text and metadata
documents_store = []

def np_normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-10)

def add_document(filename, chunks):
    global index, documents_store
    
    if not chunks:
        return
        
    embeddings = model.encode(chunks)
    embeddings = np_normalize(embeddings)
    
    start_idx = len(documents_store)
    index.add(embeddings)
    
    for i, chunk in enumerate(chunks):
        documents_store.append({
            "filename": filename,
            "text": chunk,
            "id": start_idx + i
        })

def check_duplicate(chunks, threshold=0.98):
    """
    Checks if a highly similar document already exists.
    Returns: (is_duplicate: bool, similar_filename: str)
    """
    if len(documents_store) == 0 or not chunks:
        return False, None
    
    # Check the first chunk or two to see if they're identical to existing
    check_chunks = chunks[:2]
    embeddings = model.encode(check_chunks)
    embeddings = np_normalize(embeddings)
    
    D, I = index.search(embeddings, 1)
    
    # D contains inner products (since normalized, these are cosine similarities)
    # I contains the faiss indices
    
    for i, dists in enumerate(D):
        if dists[0] > threshold:
            idx = I[i][0]
            if idx != -1 and idx < len(documents_store):
                meta = documents_store[idx]
                return True, meta["filename"]
                
    return False, None

def search(query, top_k=3):
    if len(documents_store) == 0:
        return []
        
    q_emb = model.encode([query])
    q_emb = np_normalize(q_emb)
    
    # We may want to bound top_k by len(documents_store)
    k = min(top_k, len(documents_store))
    D, I = index.search(q_emb, k)
    
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx != -1:
            results.append(documents_store[idx])
            
    return results
