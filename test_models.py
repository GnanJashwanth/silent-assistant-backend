import os, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import requests
load_dotenv()

k = os.getenv("GEMINI_API_KEY")

# List all available models via REST API
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={k}"
r = requests.get(url, timeout=15)
if r.status_code == 200:
    models = r.json().get("models", [])
    for m in models:
        name = m.get("name", "")
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            print(f"{name}")
else:
    print(f"Error listing models: {r.status_code}")
    print(r.text[:500])
