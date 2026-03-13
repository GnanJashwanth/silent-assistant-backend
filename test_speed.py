import time
print("DEBUG: Checking imports...")
start = time.time()
try:
    from sentence_transformers import SentenceTransformer
    print(f"DEBUG: Import took {time.time() - start:.2f}s")
    
    print("DEBUG: Loading model (first time might download)...")
    start = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"DEBUG: Model load took {time.time() - start:.2f}s")
    
    print("DEBUG: Testing encoding...")
    start = time.time()
    model.encode(["hello world"])
    print(f"DEBUG: Encoding took {time.time() - start:.2f}s")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
