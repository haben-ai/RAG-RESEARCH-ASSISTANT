import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_STORE_PATH


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """
    Loads the HuggingFace embedding model.
    Keeps configuration centralized.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def build_vector_store(text: str):
    """
    Build and persist FAISS vector store from raw extracted text.
    """
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)

    # 🔹 Better chunking for academic papers
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,      # larger chunks = better context for research
        chunk_overlap=150,   # smoother semantic transitions
        separators=["\n\n", "\n", ".", " ", ""]
    )

    docs = splitter.create_documents([text])

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)

    print(f"✅ Vector store built with {len(docs)} chunks")

    return len(docs)


def load_vector_store():
    """
    Load an existing FAISS vector store from disk.
    This is very fast and should be used for every question.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        raise ValueError("Vector store not found. Build it first.")

    embeddings = get_embeddings()

    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )