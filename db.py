"""PostgreSQL connection and basic CRUD operations."""

import json
import logging
import os
from typing import List, Tuple

import psycopg2
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_connection():
    """Create and return a PostgreSQL connection."""
    postgres_url = os.getenv("POSTGRES_URL")
    return psycopg2.connect(postgres_url)


def init_schema():
    """Create the documents table if it doesn't exist."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding JSON NOT NULL,
                    filename TEXT NOT NULL,
                    split_strategy TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            conn.commit()
            logger.info("Database schema initialized")
    finally:
        conn.close()


def insert_chunks(rows: List[dict]):
    """Insert document chunks into the database.
    
    Args:
        rows: List of dicts with keys: chunk_text, embedding, filename, split_strategy
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute("""
                    INSERT INTO documents (chunk_text, embedding, filename, split_strategy)
                    VALUES (%s, %s, %s, %s)
                """, (
                    row['chunk_text'],
                    json.dumps(row['embedding']),
                    row['filename'],
                    row['split_strategy']
                ))
            conn.commit()
            logger.info(f"Inserted {len(rows)} chunks")
    finally:
        conn.close()


def normalize_embedding(embedding) -> List[float]:
    """Normalize embedding to ensure it's a flat list of floats.
    
    Args:
        embedding: Embedding data from database (could be JSON string, dict, or list)
        
    Returns:
        Normalized list of float values
        
    Raises:
        ValueError: If embedding cannot be normalized
    """
    if isinstance(embedding, str):
        try:
            embedding = json.loads(embedding)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse embedding JSON")
    
    if isinstance(embedding, dict) and 'values' in embedding:
        embedding = embedding['values']
    
    if isinstance(embedding, dict) and 'embedding' in embedding:
        embedding = embedding['embedding']
        if isinstance(embedding, dict) and 'values' in embedding:
            embedding = embedding['values']
    
    if not isinstance(embedding, list):
        raise ValueError(f"Embedding is not a list: {type(embedding)}")
    
    try:
        normalized = [float(x) for x in embedding]
        return normalized
    except (ValueError, TypeError):
        raise ValueError("Embedding contains non-numeric values")


def fetch_all_chunks() -> List[Tuple[str, List[float], str, str]]:
    """Fetch all chunks and their embeddings for search.
    
    Returns:
        List of (chunk_text, embedding, filename, split_strategy) tuples
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_text, embedding, filename, split_strategy FROM documents")
            rows = cur.fetchall()
            processed_rows = []
            for text, emb, filename, split_strategy in rows:
                try:
                    normalized_embedding = normalize_embedding(emb)
                    processed_rows.append((text, normalized_embedding, filename, split_strategy))
                except ValueError as e:
                    logger.warning(f"Failed to normalize embedding for chunk: {text[:50]}... Error: {e}")
                    continue
            return processed_rows
    finally:
        conn.close()
