from pathlib import Path
import os

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# Ensure directories exist
os.makedirs(CHROMA_DIR, exist_ok=True)