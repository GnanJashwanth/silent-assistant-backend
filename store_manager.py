import numpy as np
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Absolute path for persistence to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "final_vector_state.pkl")

class PersistentStore:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            print("DEBUG: Initializing PersistentStore Singleton...")
            cls._instance = super().__new__(cls)
            cls._instance.documents_store = []
            cls._instance.faiss_index = None
            cls._instance._model = None
            # Auto-load on init
            cls._instance._load_state()
        return cls._instance

    def _get_model(self):
        if self._model is None:
            print("DEBUG: Loading local model (all-MiniLM-L6-v2)...")
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model

    def _save_state(self):
        try:
            with open(STATE_FILE, 'wb') as f:
                pickle.dump(self.documents_store, f)
            print(f"DEBUG: Saved {len(self.documents_store)} chunks to {STATE_FILE}")
        except Exception as e:
            print(f"DEBUG ERROR: Save failed: {e}")

    def _load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                print(f"DEBUG: Loading state from {STATE_FILE}")
                with open(STATE_FILE, 'rb') as f:
                    self.documents_store = pickle.load(f)
                
                if self.documents_store:
                    print(f"DEBUG: Rebuilding index for {len(self.documents_store)} chunks...")
                    model = self._get_model()
                    texts = [doc["text"] for doc in self.documents_store]
                    embeddings = model.encode(texts)
                    
                    dim = embeddings.shape[1]
                    self.faiss_index = faiss.IndexFlatIP(dim)
                    faiss.normalize_L2(embeddings)
                    self.faiss_index.add(embeddings)
                    print(f"DEBUG: Store recovered {len(self.documents_store)} chunks.")
            except Exception as e:
                print(f"DEBUG ERROR: Load failed: {e}")

    def clear_all(self):
        print("DEBUG: Clearing all documents and index for a fresh start.")
        self.documents_store = []
        self.faiss_index = None
        if os.path.exists(STATE_FILE):
            try:
                os.remove(STATE_FILE)
                print(f"DEBUG: Deleted state file {STATE_FILE}")
            except Exception as e:
                print(f"DEBUG ERROR: Could not delete state file: {e}")

    def add_document(self, filename, chunks):
        if not chunks: return
        
        # Fresh start for every new document as requested
        self.clear_all()
        
        model = self._get_model()
        print(f"DEBUG: Processing {len(chunks)} new chunks for {filename}...")
        embeddings = model.encode(chunks)
        faiss.normalize_L2(embeddings)
        
        # Index setup
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        
        for i, chunk in enumerate(chunks):
            self.documents_store.append({
                "filename": filename,
                "text": chunk,
                "id": i
            })
            
        self._save_state()
        print(f"DEBUG: Store updated with {filename} (Total: {len(self.documents_store)} chunks)")

    def search(self, query, top_k=5):
        if not self.documents_store or self.faiss_index is None:
            print("DEBUG: Search aborted - store is empty.")
            return []
            
        print(f"DEBUG: Searching for '{query}'...")
        model = self._get_model()
        q_emb = model.encode([query])
        faiss.normalize_L2(q_emb)
        
        limit = min(top_k, len(self.documents_store))
        D, I = self.faiss_index.search(q_emb, limit)
        
        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.documents_store):
                results.append(self.documents_store[idx])
        return results

# Export a single instance
store = PersistentStore()
