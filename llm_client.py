import os
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Gemini REST API
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-004"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
EMBEDDING_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:batchEmbedContents"

def get_embeddings(texts: list) -> list:
    """
    Calls Google Gemini Embedding API to get vectors for a list of strings.
    """
    if not api_key:
        print("Error: GEMINI_API_KEY not set.")
        return []

    headers = {"Content-Type": "application/json"}
    
    # Prepare batch request
    print(f"Requesting embeddings for {len(texts)} chunks...")
    requests_list = []
    for text in texts:
        requests_list.append({
            "model": f"models/{EMBEDDING_MODEL}",
            "content": {"parts": [{"text": text}]}
        })
    
    payload = {"requests": requests_list}
    
    try:
        response = requests.post(
            f"{EMBEDDING_API_URL}?key={api_key}",
            json=payload,
            headers=headers,
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            # Extract embeddings from batch response
            embeddings = [item["values"] for item in data["embeddings"]]
            print(f"Successfully retrieved {len(embeddings)} embeddings.")
            return embeddings
        else:
            print(f"Embedding API Error ({response.status_code}): {response.text}")
            return []
    except Exception as e:
        print(f"Embedding Connection Error: {str(e)}")
        return []

def generate_answer(query: str, context_chunks: list) -> str:
    """
    Calls Google Gemini REST API with ultra-strict prompt for 100% exactness.
    """
    context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])
    
    prompt = f"""You are the "Silent Assistant", a high-precision document extraction engine.
Your mission: Find the EXACT string requested in the USER QUERY from the provided PDF DATA.

STRICT PROTOCOLS:
1. RESPONSE FORMAT: Output ONLY the requested information. NO preamble, NO "The title is...", NO "Based on the context...".
2. If the user asks for a TITLE, look for the first line or a name near "PROJECT REPORT".
3. If the user asks for a NAME, find the exact spelling from the text.
4. If the info is NOT present, output: "DATA_NOT_FOUND".
5. VERBATIM ONLY: Use the exact characters, casing, and punctuation from the document.

PDF DATA FOR EXTRACTION:
{context_text}

USER QUERY: {query}

SILENT ASSISTANT RESPONSE:"""
    
    if not api_key:
        return "Error: GEMINI_API_KEY not set in .env file."

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    # Retry up to 3 times with increasing delays for quota errors
    for attempt in range(3):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={api_key}",
                json=payload,
                headers=headers,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif response.status_code == 429:
                # Quota exceeded - wait and retry
                wait_time = (attempt + 1) * 15  # 15s, 30s, 45s
                if attempt < 2:
                    print(f"[Retry] Quota limit hit, waiting {wait_time}s before retry {attempt+2}/3...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "Gemini API quota temporarily exceeded. Please wait 1-2 minutes and try again."
            else:
                error_msg = response.json().get("error", {}).get("message", response.text[:200])
                return f"Gemini API Error ({response.status_code}): {error_msg}"
        except Exception as e:
            return f"Connection Error: {str(e)}"
    
    return "Unable to get a response after multiple retries. Please wait a moment and try again."
