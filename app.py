from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import document_processor
import vector_store
import llm_client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Silent Assistant")

# Add CORS so frontend can communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Extract text
        text = document_processor.process_document(file.filename, contents)
        
        # Chunk text
        chunks = document_processor.chunk_text(text)
        
        # Check duplicate
        is_duplicate, similar_file = vector_store.check_duplicate(chunks)
        
        # Store embeddings
        vector_store.add_document(file.filename, chunks)
        
        return {
            "message": f"Successfully processed {file.filename}.",
            "chunks_processed": len(chunks),
            "duplicate_warning": is_duplicate,
            "similar_file": similar_file
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(query: str = Form(...)):
    try:
        # 1. Get semantic results
        semantic_chunks = vector_store.search(query, top_k=8)
        
        # 2. Always include first 2 chunks (for titles/metadata)
        first_chunks = vector_store.documents_store[:2]
        
        # 3. Combine and remove duplicates
        seen_ids = set()
        context_chunks = []
        for c in (first_chunks + semantic_chunks):
            if c["id"] not in seen_ids:
                context_chunks.append(c)
                seen_ids.add(c["id"])
        
        if not context_chunks:
            return {"answer": "No documents available. Please upload a document first.", "context": []}
            
        # Generate Answer
        answer = llm_client.generate_answer(query, context_chunks)
        
        return {
            "answer": answer,
            "context": [c["text"] for c in context_chunks]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
