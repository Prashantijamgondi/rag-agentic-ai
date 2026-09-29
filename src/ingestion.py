from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
from pinecone import Pinecone, ServerlessSpec
import os

def run_ingestion(pdf_path: str, index_name: str):
    print(f"Loading PDF from {pdf_path}...")
    # 1. Load document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(f"Loaded {len(docs)} pages. Splitting text...")
    # 2. Chunk document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # Ensure index exists
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Index created.")

    print("Creating embeddings and saving to Pinecone...")
    # 3. Create embeddings & save to Pinecone
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=PINECONE_API_KEY
    )
    print("Ingestion complete!")
    return vector_store

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_file_path = os.path.join(base_dir, "data", "Ebook-Agentic-AI.pdf")
    run_ingestion(pdf_file_path, PINECONE_INDEX_NAME)
