import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = BASE_DIR / '.models'
HF_CACHE_DIR = MODEL_CACHE_DIR / 'huggingface'
CHROMA_DIR = BASE_DIR / 'chroma_db'
PUBLIC_DEMO_MODE = os.getenv('PUBLIC_DEMO_MODE', '').strip().lower() in {'1', 'true', 'yes'}
EMBEDDING_MODEL = os.getenv('LOCAL_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
OLLAMA_TIMEOUT_SECONDS = float(os.getenv('OLLAMA_TIMEOUT_SECONDS', '120'))
MAX_DOCUMENTS = 8 if PUBLIC_DEMO_MODE else 100
MAX_DOCUMENT_CHARS = 10000 if PUBLIC_DEMO_MODE else 1000000

os.environ['HF_HOME'] = str(HF_CACHE_DIR)
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
