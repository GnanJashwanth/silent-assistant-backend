import os, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import requests
load_dotenv()

k = os.getenv("GEMINI_API_KEY")
# Try models that usually have higher free tier limits or are less likely to be exhausted
models = ["gemini-1.5-flash-8b", "gemini-1.5-flash", "gemini-1.5-pro"]

for m in models:
    # Try both with and without models/ prefix
    for prefix in ["models/", ""]:
        full_name = f"{prefix}{m}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{full_name}:generateContent?key={k}"
        try:
            r = requests.post(url, json={"contents": [{"parts": [{"text": "Say hi"}]}]}, timeout=10)
            print(f"{full_name}: {r.status_code}")
            if r.status_code == 200:
                print(f"  WORKS! -> {full_name}")
                # Save working model to a temp file for the next step
                with open("working_model.txt", "w") as f:
                    f.write(full_name)
                exit(0)
        except:
            continue
