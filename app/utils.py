import os

DATA_DIR = "data"
INDEX_DIR = "models/faiss_index"

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
