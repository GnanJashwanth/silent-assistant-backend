import os, requests, json
from dotenv import load_dotenv
load_dotenv()
k = os.getenv('GEMINI_API_KEY')
r = requests.get(f'https://generativelanguage.googleapis.com/v1beta/models?key={k}')
models = r.json().get('models', [])
flash_models = [m['name'] for m in models if 'flash' in m['name'].lower()]
print(json.dumps(flash_models, indent=2))
