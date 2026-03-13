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
        # Ensure directory exists (relevant for /tmp or custom paths)
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, 'wb') as f:
            pickle.dump((documents_store, vector_store), f)
        print(f"Successfully saved {len(documents_store)} chunks to {STATE_FILE}")
    except Exception as e:
        print(f"CRITICAL: Failed to save state to {STATE_FILE}: {e}")

def _load_state():
    global documents_store, vector_store
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                loaded_docs, loaded_vectors = pickle.load(f)
                
                # In-place update to keep the same list object reference for other modules
                documents_store.clear()
                documents_store.extend(loaded_docs)
                vector_store = loaded_vectors
                
            print(f"Successfully loaded {len(documents_store)} chunks from {STATE_FILE}")
        except Exception as e:
            print(f"CRITICAL: Failed to load state from {STATE_FILE}: {e}")
    else:
        print(f"State file NOT FOUND at {STATE_FILE}")

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
        print("Warning: No chunks to add.")
        return
        
    print(f"Getting embeddings for {len(chunks)} chunks of {filename}...")
    embeddings = llm_client.get_embeddings(chunks)
    if not embeddings:
        print("CRITICAL: Final embeddings list is empty. API error?")
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
    
    print(f"Processed {len(chunks)} chunks. Total in store: {len(documents_store)}")
    _save_state()

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
        print("Search failed: documents_store or vector_store is empty.")
        return []
        
    q_emb = llm_client.get_embeddings([query])
    if not q_emb:
        print("Search failed: Could not get embedding for query.")
        return []
        
    q_emb = np_normalize(q_emb)
    similarities = np.dot(q_emb, vector_store.T)[0]
    
    k = min(top_k, len(documents_store))
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append(documents_store[idx])
            
    print(f"Search returned {len(results)} matches for query: '{query}'")
    return results
