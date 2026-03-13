import numpy as np
import llm_client
import pickle
import os

# Local persistence file
STATE_FILE = "vector_state.pkl"

# Global data containers
documents_store = []
vector_store = None

def _save_state():
    """Saves current memory state to a local file."""
    global documents_store, vector_store
    try:
        with open(STATE_FILE, 'wb') as f:
            pickle.dump((documents_store, vector_store), f)
        print(f"DEBUG: Saved {len(documents_store)} chunks locally.")
    except Exception as e:
        print(f"DEBUG ERROR: Save failed: {e}")

def _load_state():
    """Loads state from local file into memory."""
    global documents_store, vector_store
    
    # If already loaded in memory, don't reload unless empty
    if len(documents_store) > 0:
        return

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                loaded_docs, loaded_vectors = pickle.load(f)
                documents_store.clear()
                documents_store.extend(loaded_docs)
                vector_store = loaded_vectors
            print(f"DEBUG: Loaded {len(documents_store)} chunks from disk.")
        except Exception as e:
            print(f"DEBUG ERROR: Load failed: {e}")

def np_normalize(vecs):
    v = np.array(vecs)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-10)

def add_document(filename, chunks):
    global documents_store, vector_store
    
    if not chunks:
        print("DEBUG: No text found in document.")
        return
        
    print(f"DEBUG: Getting embeddings for {len(chunks)} chunks...")
    embeddings = llm_client.get_embeddings(chunks)
    if not embeddings:
        print("DEBUG ERROR: API failed to return embeddings.")
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
    
    print(f"DEBUG: Successfully added {filename}. Total: {len(documents_store)}")
    _save_state()

def check_duplicate(chunks, threshold=0.98):
    _load_state()
    if len(documents_store) == 0 or not chunks or vector_store is None:
        return False, None
    
    embeddings = llm_client.get_embeddings(chunks[:2])
    if not embeddings: return False, None
        
    embeddings = np_normalize(embeddings)
    similarities = np.dot(embeddings, vector_store.T)
    
    for i in range(len(embeddings)):
        max_idx = np.argmax(similarities[i])
        if similarities[i][max_idx] > threshold:
            return True, documents_store[max_idx]["filename"]
                
    return False, None

def search(query, top_k=5):
    _load_state()
    global vector_store
    
    if len(documents_store) == 0 or vector_store is None:
        print("DEBUG: Search failed - Store is empty.")
        return []
        
    print(f"DEBUG: Searching for: {query}")
    q_emb = llm_client.get_embeddings([query])
    if not q_emb: return []
        
    q_emb = np_normalize(q_emb)
    similarities = np.dot(q_emb, vector_store.T)[0]
    
    k = min(top_k, len(documents_store))
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = [documents_store[idx] for idx in top_indices]
    print(f"DEBUG: Found {len(results)} matches.")
    return results
