"""Index documents into the RAG system."""

import argparse
import logging
import os
from pathlib import Path
from typing import List

import PyPDF2
from docx import Document
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

import db
import text_utils

load_dotenv(find_dotenv(), override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from a PDF file using pdfplumber with PyPDF2 fallback."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            logger.info("Successfully extracted text using pdfplumber")
            return text
        else:
            logger.warning("pdfplumber extracted empty text, falling back to PyPDF2")
    except ImportError:
        logger.info("pdfplumber not available, using PyPDF2")
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}, falling back to PyPDF2")
    
    try:
        text = ""
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logger.info("Successfully extracted text using PyPDF2")
        return text
    except Exception as e:
        logger.error(f"Both pdfplumber and PyPDF2 failed to extract text: {e}")
        raise ValueError(f"Failed to extract text from PDF: {e}")


def extract_text_from_docx(filepath: str) -> str:
    """Extract text from a DOCX file."""
    doc = Document(filepath)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text


def extract_text(filepath: str) -> str:
    """Extract text from PDF or DOCX file."""
    ext = Path(filepath).suffix.lower()
    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext == '.docx':
        return extract_text_from_docx(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using Google Gemini.
    
    Args:
        texts: List of text chunks to embed
    
    Returns:
        List of embedding vectors
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    genai.configure(api_key=api_key)
    
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        # Extract embedding values from result
        if isinstance(result, dict) and 'embedding' in result:
            embedding = result['embedding']
            if isinstance(embedding, dict) and 'values' in embedding:
                embeddings.append(embedding['values'])
            elif isinstance(embedding, list):
                embeddings.append(embedding)
        else:
            raise ValueError(f"Unexpected embedding result format: {type(result)}")
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Index documents into RAG system")
    parser.add_argument("--file", required=True, help="Path to PDF or DOCX file")
    parser.add_argument("--split_strategy", default="fixed-with-overlap",
                       choices=["fixed-with-overlap", "sentence", "paragraph"],
                       help="Text splitting strategy")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return
    
    db.init_schema()
    
    logger.info(f"Extracting text from {args.file}")
    text = extract_text(args.file)
    
    if not text.strip():
        logger.error("No text extracted from file")
        return
    
    logger.info(f"Splitting text using strategy: {args.split_strategy}")
    chunks = text_utils.split_text(text, args.split_strategy)
    logger.info(f"Created {len(chunks)} chunks")
    
    logger.info("Generating embeddings...")
    embeddings = get_embeddings(chunks)
    
    filename = os.path.basename(args.file)
    rows = [
        {
            'chunk_text': chunk,
            'embedding': embedding,
            'filename': filename,
            'split_strategy': args.split_strategy
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]
    
    db.insert_chunks(rows)
    logger.info(f"Successfully indexed {filename}")


if __name__ == "__main__":
    main()
