"""Search indexed documents using semantic similarity."""

import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv, find_dotenv

import db

load_dotenv(find_dotenv(), override=True)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)


def extract_embedding_values(result) -> List[float]:
    """Extract embedding values from Gemini API response.
    
    Args:
        result: Response from genai.embed_content
        
    Returns:
        Flat list of float values
        
    Raises:
        ValueError: If embedding cannot be extracted
    """
    if isinstance(result, dict) and 'embedding' in result:
        embedding = result['embedding']
        if isinstance(embedding, dict) and 'values' in embedding:
            return embedding['values']
        elif isinstance(embedding, list):
            return embedding
    
    if isinstance(result, dict) and 'data' in result:
        if isinstance(result['data'], list) and len(result['data']) > 0:
            data_item = result['data'][0]
            if isinstance(data_item, dict) and 'embedding' in data_item:
                embedding = data_item['embedding']
                if isinstance(embedding, dict) and 'values' in embedding:
                    return embedding['values']
                elif isinstance(embedding, list):
                    return embedding
    
    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], (int, float)):
            return result
        elif len(result) > 0 and isinstance(result[0], dict):
            first_item = result[0]
            if 'embedding' in first_item:
                embedding = first_item['embedding']
                if isinstance(embedding, dict) and 'values' in embedding:
                    return embedding['values']
                elif isinstance(embedding, list):
                    return embedding
    
    raise ValueError(f"Could not extract embedding values from result: {type(result)}")


def get_query_embedding(query: str) -> List[float]:
    """Get embedding for search query using Google Gemini."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    genai.configure(api_key=api_key)
    
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query
    )
    return extract_embedding_values(result)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def search(query: str, k: int = 5) -> List[Tuple[str, float, str, str]]:
    """Search for top k most similar chunks to the query.
    
    Args:
        query: Search query text
        k: Number of top results to return
    
    Returns:
        List of (chunk_text, similarity_score, filename, split_strategy) tuples
    """
    logger.info("Generating query embedding...")
    query_embedding = get_query_embedding(query)
    
    logger.info("Fetching chunks from database...")
    chunks = db.fetch_all_chunks()
    
    if not chunks:
        logger.warning("No documents found in database")
        return []
    
    logger.info("Computing similarities...")
    similarities = []
    for chunk_text, chunk_embedding, filename, split_strategy in chunks:
        sim = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((chunk_text, sim, filename, split_strategy))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def main():
    parser = argparse.ArgumentParser(description="Search indexed documents")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()
    
    results = search(args.query, args.k)
    
    if not results:
        print(json.dumps([], ensure_ascii=False, indent=2))
        return
    
    results_list = []
    for i, (chunk, score, filename, split_strategy) in enumerate(results, 1):
        results_list.append({
            'rank': i,
            'similarity': float(score),
            'file': filename,
            'split_strategy': split_strategy,
            'chunk_text': chunk
        })
    
    print(json.dumps(results_list, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
