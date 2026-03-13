import numpy as np
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Absolute path for persistence to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "vector_state_local.pkl")

# Global model and indices
_model = None
documents_store = []
faiss_index = None

def get_model():
    global _model
    if _model is None:
        print("DEBUG: Loading local embedding model (offline)...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def _save_state():
    global documents_store
    try:
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(documents_store, f)
        print(f"DEBUG: Successfully saved {len(documents_store)} chunks to {STATE_FILE}.")
    except Exception as e:
        print(f"DEBUG ERROR: Save failed to {STATE_FILE}: {e}")

def _load_state():
    global documents_store, faiss_index
    # If already loaded in memory AND index exists, we are good
    if len(documents_store) > 0 and faiss_index is not None:
        return

    if os.path.exists(STATE_FILE):
        try:
            print(f"DEBUG: Loading state from {STATE_FILE}...")
            with open(STATE_FILE, 'rb') as f:
                loaded_docs = pickle.load(f)
                
                # In-place update to preserve references in other modules (like app.py)
                documents_store.clear()
                documents_store.extend(loaded_docs)
            
            # Rebuild index if we have documents
            if documents_store:
                print(f"DEBUG: Rebuilding index for {len(documents_store)} chunks...")
                model = get_model()
                texts = [doc["text"] for doc in documents_store]
                embeddings = model.encode(texts)
                
                dim = embeddings.shape[1]
                faiss_index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(embeddings)
                faiss_index.add(embeddings)
                print(f"DEBUG: Index rebuilt successfully with {faiss_index.ntotal} vectors.")
            else:
                print("DEBUG: State file was empty.")
        except Exception as e:
            print(f"DEBUG ERROR: Load failed from {STATE_FILE}: {e}")
    else:
        print(f"DEBUG: No persistence file at {STATE_FILE}. Starting with empty store.")

def add_document(filename, chunks):
    global documents_store, faiss_index
    
    # Reload existing documents first
    _load_state()
    
    if not chunks:
        print("DEBUG: Warning - No text found in document.")
        return
        
    model = get_model()
    print(f"DEBUG: Generating embeddings for {len(chunks)} chunks of {filename}...")
    embeddings = model.encode(chunks)
    
    faiss.normalize_L2(embeddings)
    
    start_idx = len(documents_store)
    
    if faiss_index is None:
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        
    faiss_index.add(embeddings)
    
    for i, chunk in enumerate(chunks):
        documents_store.append({
            "filename": filename,
            "text": chunk,
            "id": start_idx + i
        })
    
    print(f"DEBUG: Added {filename}. Current memory count: {len(documents_store)}")
    _save_state()

def check_duplicate(chunks, threshold=0.98):
    _load_state()
    if len(documents_store) == 0 or not chunks or faiss_index is None:
        return False, None
    
    model = get_model()
    check_chunks = chunks[:2]
    embeddings = model.encode(check_chunks)
    faiss.normalize_L2(embeddings)
    
    D, I = faiss_index.search(embeddings, 1)
    
    for i, dist in enumerate(D):
        if dist[0] > threshold:
            idx = I[i][0]
            if idx != -1 and idx < len(documents_store):
                return True, documents_store[idx]["filename"]
                
    return False, None

def search(query, top_k=5):
    # Ensure memory is populated
    _load_state()
    global faiss_index
    
    if len(documents_store) == 0:
        print("DEBUG: Search aborted - documents_store is empty.")
        return []
    
    if faiss_index is None:
        print("DEBUG: Search aborted - faiss_index is None.")
        return []
        
    print(f"DEBUG: Searching locally for: '{query}'")
    model = get_model()
    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)
    
    num_to_search = min(top_k, len(documents_store))
    D, I = faiss_index.search(q_emb, num_to_search)
    
    results = []
    for idx in I[0]:
        if idx != -1 and idx < len(documents_store):
            results.append(documents_store[idx])
            
    print(f"DEBUG: Found {len(results)} relevant chunks.")
    return results
