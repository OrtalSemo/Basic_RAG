"""Text splitting utilities."""

import re
from typing import List


def split_fixed_overlap(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into fixed-size chunks with overlap.
    
    Args:
        text: Input text to split
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += chunk_size - overlap
    
    return chunks


def split_sentence(text: str) -> List[str]:
    """Split text by sentences using simple regex.
    
    Args:
        text: Input text to split
    
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_paragraph(text: str) -> List[str]:
    """Split text by paragraphs (blank lines).
    
    Args:
        text: Input text to split
    
    Returns:
        List of paragraphs
    """
    if not text:
        return []
    
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_text(text: str, strategy: str = "fixed-with-overlap") -> List[str]:
    """Split text using the specified strategy.
    
    Args:
        text: Input text to split
        strategy: One of 'fixed-with-overlap', 'sentence', 'paragraph'
    
    Returns:
        List of text chunks
    """
    if strategy == "fixed-with-overlap":
        return split_fixed_overlap(text)
    elif strategy == "sentence":
        return split_sentence(text)
    elif strategy == "paragraph":
        return split_paragraph(text)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
