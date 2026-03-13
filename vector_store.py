import numpy as np
import llm_client

# Maps indices to chunk text and metadata
documents_store = []
# Holds the embedding vectors
vector_store = None

def np_normalize(vecs):
    # vecs can be a list of lists or a 2D numpy array
    v = np.array(vecs)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-10)

def add_document(filename, chunks):
    global documents_store, vector_store
    
    if not chunks:
        return
        
    # Get embeddings from Gemini API
    embeddings = llm_client.get_embeddings(chunks)
    if not embeddings:
        print(f"Failed to get embeddings for {filename}")
        return
        
    embeddings = np_normalize(embeddings)
    
    start_idx = len(documents_store)
    
    if vector_store is None:
        vector_store = embeddings
    else:
        vector_store = np.vstack([vector_store, embeddings])
    
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
    global vector_store
    
    if len(documents_store) == 0 or not chunks or vector_store is None:
        return False, None
    
    # Check the first chunk or two to see if they're identical to existing
    check_chunks = chunks[:2]
    embeddings = llm_client.get_embeddings(check_chunks)
    if not embeddings:
        return False, None
        
    embeddings = np_normalize(embeddings)
    
    # Compute cosine similarity: dot product of normalized vectors
    # vector_store is (N, D), embeddings is (M, D)
    # similarities is (M, N)
    similarities = np.dot(embeddings, vector_store.T)
    
    for i in range(len(embeddings)):
        max_sim_idx = np.argmax(similarities[i])
        if similarities[i][max_sim_idx] > threshold:
            meta = documents_store[max_sim_idx]
            return True, meta["filename"]
                
    return False, None

def search(query, top_k=3):
    global vector_store
    
    if len(documents_store) == 0 or vector_store is None:
        return []
        
    q_emb = llm_client.get_embeddings([query])
    if not q_emb:
        return []
        
    q_emb = np_normalize(q_emb)
    
    # similarities shape: (1, N)
    similarities = np.dot(q_emb, vector_store.T)[0]
    
    # Get top k indices
    k = min(top_k, len(documents_store))
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append(documents_store[idx])
            
    return results
