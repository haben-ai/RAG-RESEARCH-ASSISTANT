# rag_engine.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from config import VECTOR_STORE_PATH

# Initialize Gemini client with API key from environment variable
client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

# --- VECTOR STORE FUNCTIONS ---

def build_vector_store(text):
    """
    Build a FAISS vector store from a text string.
    Saves the vector store locally to VECTOR_STORE_PATH.
    """
    os.makedirs("vectorstore", exist_ok=True)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    docs = splitter.create_documents([text])

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)

    return len(docs)


def load_vector_store():
    """
    Load the FAISS vector store from disk.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# --- RAG QUESTION ANSWERING FUNCTION ---

def answer_question(question, vectorstore, top_k=3):
    """
    Retrieve relevant documents from vector store and answer a question
    using Gemini API.

    Args:
        question (str): The user's question.
        vectorstore (FAISS): Loaded FAISS vector store.
        top_k (int): Number of documents to retrieve.

    Returns:
        str: Answer text from Gemini.
    """
    # Retrieve top-k relevant docs
    docs = vectorstore.similarity_search(question, k=top_k)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Call Gemini API with context
    response = client.responses.create(
        model="gemini-2.5-flash",  # Choose a valid model from list_models()
        input=f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
    )

    # Return answer text
    return response.output[0].content[0].text