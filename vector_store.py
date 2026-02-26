from langchain.text_splitter import RecursiveCharacterTextSplitter  # correct for your version
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # updated from deprecated langchain_community
from config import VECTOR_STORE_PATH
import os

def build_vector_store(text):
    os.makedirs("vectorstore", exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)

    return len(docs)


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )