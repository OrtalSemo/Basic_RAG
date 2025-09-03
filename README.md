# RAG Pipeline with Gemini API

A Retrieval-Augmented Generation (RAG) pipeline using Google Gemini embeddings and PostgreSQL for semantic search.

## Overview

This project extracts text from documents, splits it into chunks, generates embeddings with Gemini, and stores them in PostgreSQL for fast semantic search.

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
POSTGRES_URL=postgresql://username:password@localhost:5432/database_name
```

3. **Ensure PostgreSQL is running** and accessible with the configured connection string.

## Usage

### Indexing Documents

Index a document to make it searchable:

```bash
# Index a PDF file
python index_documents.py --file document.pdf

# Index a DOCX file with custom splitting strategy
python index_documents.py --file document.docx --split_strategy sentence

# Index with paragraph-based splitting
python index_documents.py --file document.pdf --split_strategy paragraph
```

**Available split strategies:**
- `fixed-with-overlap` (default): 800-character chunks with 100-character overlap
- `sentence`: Split on sentence boundaries using punctuation
- `paragraph`: Split on blank lines between paragraphs

### Searching Documents

Search indexed documents using natural language queries:

```bash
# Basic search
python search_documents.py --query "machine learning algorithms"

# Search with custom result count
python search_documents.py --query "neural networks" --k 5

# Search for specific concepts
python search_documents.py --query "deep learning applications in healthcare"
```

**Search parameters:**
- `--query`: Your search query (required)
- `--k`: Number of top results to return (default: 5)

## Project Structure

- **`index_documents.py`**: Main script for indexing documents into the system
- **`search_documents.py`**: Main script for searching indexed documents
- **`db.py`**: Database operations and PostgreSQL connection management
- **`text_utils.py`**: Text splitting utilities and strategies

## Database Schema

The system automatically creates a `documents` table with:
- `id`: Unique identifier
- `chunk_text`: Text content of the document chunk
- `embedding`: Vector embedding stored as JSON
- `filename`: Source document filename
- `split_strategy`: Text splitting method used
- `created_at`: Timestamp of indexing

## Requirements

- Python 3.7+
- PostgreSQL database
- Google Gemini API key
- Dependencies listed in `requirements.txt`
