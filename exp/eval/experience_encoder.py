# experience_encoder.py
import torch
from sentence_transformers import SentenceTransformer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_MODEL = None
_MODEL_PATH = "your_bge-m3_model_path"

def get_encoder():
    """
    Globally unique SentenceTransformer
    Loaded on the first call and reused thereafter.
    """
    global _MODEL
    if _MODEL is None:
        print("🚀 Loading experience encoder once...")
        _MODEL = SentenceTransformer(
            _MODEL_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    return _MODEL
