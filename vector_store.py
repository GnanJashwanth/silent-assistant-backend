import numpy as np
import llm_client
import pickle
import os

# Persistence path - relative to the script for better local compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "vector_state.pkl")

# Maps indices to chunk text and metadata
documents_store = []
# Holds the embedding vectors
vector_store = None

def _save_state():
    global documents_store, vector_store
    try:
        with open(STATE_FILE, 'wb') as f:
            pickle.dump((documents_store, vector_store), f)
    except Exception as e:
        print(f"Failed to save state: {e}")

def _load_state():
    global documents_store, vector_store
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                documents_store, vector_store = pickle.load(f)
            print(f"Loaded state from {STATE_FILE}: {len(documents_store)} chunks")
        except Exception as e:
            print(f"Failed to load state from {STATE_FILE}: {e}")
    else:
        print(f"State file not found at {STATE_FILE}")

def np_normalize(vecs):
    v = np.array(vecs)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-10)

def add_document(filename, chunks):
    global documents_store, vector_store
    _load_state()
    
    if not chunks:
        return
        
    embeddings = llm_client.get_embeddings(chunks)
    if not embeddings:
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
    print(f"Processed embeddings for {filename}. Total chunks: {len(documents_store)}")
    
    _save_state()
    print(f"Saved state for {filename}")

def check_duplicate(chunks, threshold=0.98):
    _load_state()
    global vector_store
    
    if len(documents_store) == 0 or not chunks or vector_store is None:
        return False, None
    
    embeddings = llm_client.get_embeddings(chunks[:2])
    if not embeddings:
        return False, None
        
    embeddings = np_normalize(embeddings)
    similarities = np.dot(embeddings, vector_store.T)
    
    for i in range(len(embeddings)):
        max_sim_idx = np.argmax(similarities[i])
        if similarities[i][max_sim_idx] > threshold:
            meta = documents_store[max_sim_idx]
            return True, meta["filename"]
                
    return False, None

def search(query, top_k=3):
    _load_state()
    global vector_store
    
    if len(documents_store) == 0 or vector_store is None:
        return []
        
    q_emb = llm_client.get_embeddings([query])
    if not q_emb:
        return []
        
    q_emb = np_normalize(q_emb)
    similarities = np.dot(q_emb, vector_store.T)[0]
    
    k = min(top_k, len(documents_store))
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append(documents_store[idx])
            
    return results
