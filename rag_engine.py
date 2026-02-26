# rag_engine.py

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from config import VECTOR_STORE_PATH

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

# Build vector store
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

# Load vector store
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# Answer question
def answer_question(question, vectorstore, top_k=3):
    docs = vectorstore.similarity_search(question, k=top_k)
    context = "\n\n".join([doc.page_content for doc in docs])

    response = client.responses.create(
        model="models/gemini-2.5-flash",   # valid model name
        input=f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
    )

    # Extract text
    answer = ""
    for block in response.output:
        for element in block.content:
            if element.type == "output_text":
                answer += element.text

    return answer.strip()