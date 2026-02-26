import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from config import VECTOR_STORE_PATH

# Initialize Gemini Client
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("Please set GENAI_API_KEY in your environment variables")

client = genai.Client(api_key=GENAI_API_KEY)

# -----------------------
# Build vector store
# -----------------------
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    return len(docs)

# -----------------------
# Load vector store
# -----------------------
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# -----------------------
# Answer question
# -----------------------
def answer_question(question, vectorstore):
    # Retrieve top 3 similar chunks
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    # -----------------------
    # Generate response using latest GenAI API
    # -----------------------
    response = client.chat(
        model="chat-bison-001",  # Gemini chat model
        messages=[
            {
                "role": "system",
                "content": "You are a helpful research assistant. Use the provided context to answer questions."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ],
        temperature=0.2,
        max_output_tokens=500
    )

    return response.last["content"]