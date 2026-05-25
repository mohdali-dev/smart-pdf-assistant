import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# Initialize the model once to avoid reloading it every time
# 'all-MiniLM-L6-v2' is fast, lightweight, and great for semantic search
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def create_embeddings(chunks: List[str]) -> np.ndarray:
    """
    Converts a list of text chunks into vector embeddings.
    """
    embeddings = model.encode(chunks)
    return np.array(embeddings).astype('float32')

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Creates a FAISS index from embeddings.
    Uses L2 distance (Euclidean) which is standard for MiniLM embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_vectors(index: faiss.IndexFlatL2, query: str, k: int = 3) -> List[int]:
    """
    Searches the FAISS index for the top-k most similar vectors to the query.
    Returns the indices of the most relevant chunks.
    """
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Search returns distances and indices
    distances, indices = index.search(query_embedding, k)
    return indices[0].tolist()