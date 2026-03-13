from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

import document_processor
from store_manager import store
import llm_client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Silent Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/debug-state")
async def debug_state():
    return {
        "memory_docs_count": len(store.documents_store),
        "model_ready": store.model_ready,
        "index_ready": store.faiss_index is not None,
        "index_count": store.faiss_index.ntotal if store.faiss_index else 0,
        "state_file_disk": os.path.exists(os.path.join(os.path.dirname(__file__), "final_vector_state.pkl")),
        "cwd": os.getcwd()
    }

@app.get("/")
async def root():
    return {"message": "Silent Assistant Backend is active."}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = document_processor.process_document(file.filename, contents)
        chunks = document_processor.chunk_text(text)
        
        # Store using singleton
        store.add_document(file.filename, chunks)
        
        return {
            "message": f"Successfully processed {file.filename}.",
            "chunks_processed": len(chunks),
            "total_in_memory": len(store.documents_store)
        }
    except Exception as e:
        print(f"UPLOAD ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(query: str = Form(...)):
    try:
        # Get results from singleton
        semantic_chunks = store.search(query, top_k=8)
        
        # Merge with first 2 chunks for context
        first_chunks = store.documents_store[:2]
        
        seen_ids = set()
        context_chunks = []
        for c in (first_chunks + semantic_chunks):
            if c["id"] not in seen_ids:
                context_chunks.append(c)
                seen_ids.add(c["id"])
        
        if not context_chunks:
            return {
                "answer": "No documents found in my memory. Please (re)upload a document.", 
                "context": [],
                "debug_docs_count": len(store.documents_store)
            }
            
        answer = llm_client.generate_answer(query, context_chunks)
        
        return {
            "answer": answer,
            "context": [c["text"] for c in context_chunks],
            "debug": {
                "total_docs": len(store.documents_store),
                "context_size": len(context_chunks)
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
