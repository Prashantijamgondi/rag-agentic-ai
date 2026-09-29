# LangGraph & Pinecone RAG Chatbot

This repository contains a robust Retrieval-Augmented Generation (RAG) chatbot using LangGraph, Pinecone, and vector embeddings, serving the "Agentic AI" eBook as its knowledge base.

## Architecture
- **Data Ingestion**: PyPDF for loading, LangChain TextSplitter for chunking.
- **Embeddings & Vector DB**: OpenAI embeddings (`text-embedding-3-small`) and Pinecone for vector storage.
- **Orchestration**: LangGraph state graph with retrieve and generate nodes.
- **API Interface**: FastAPI to expose the endpoint returning structured JSON payload.

## Setup Guide

### 1. Clone & Environment Setup
```bash
git clone <your-repo-link>
cd rag-agentic-ai
python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```
Required keys:
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (default is `agentic-ai-index`)

### 3. Data Ingestion
The document is downloaded in `data/Ebook-Agentic-AI.pdf`.
Run the ingestion script to chunk, embed, and store in Pinecone:
```bash
python -m src.ingestion
```

### 4. Run the API
Start the FastAPI application:
```bash
uvicorn app:app --reload
```
The API will be available at `http://localhost:8000`. You can access the Swagger UI at `http://localhost:8000/docs`.

### 5. Testing
Run the sample test script to test against the required benchmark queries:
```bash
python tests_sample_queries.py
```
