import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

DB_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """
    Loads the embedding model.
    This should only initialize once per run.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def build_vector_store(pdf_path: str):
    """
    Build FAISS index from a PDF and save it locally.
    Only call this when processing a NEW paper.
    """
    print("📄 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    print("🧠 Creating embeddings...")
    embeddings = get_embeddings()

    print("📦 Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    print("💾 Saving FAISS index locally...")
    vectorstore.save_local(DB_PATH)

    print("✅ Vector store built successfully.")
    return vectorstore


def load_vector_store():
    """
    Load existing FAISS index from disk.
    This is FAST and should be used for every question.
    """
    if not os.path.exists(DB_PATH):
        raise ValueError("Vector store not found. Build it first.")

    embeddings = get_embeddings()

    print("⚡ Loading FAISS index from disk...")
    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore