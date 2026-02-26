import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from config import VECTOR_STORE_PATH

# -----------------------
# Initialize Gemini Client
# -----------------------
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("Please set GENAI_API_KEY in your environment variables")

# The new google-genai SDK uses this Client structure
client = genai.Client(api_key=GENAI_API_KEY)

# -----------------------
# Build vector store
# -----------------------
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    docs = splitter.create_documents([text])

    # Using HuggingFace for local embeddings to save on API costs
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    return len(docs)

# -----------------------
# Load vector store
# -----------------------
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # allow_dangerous_deserialization is required for loading local FAISS files
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# -----------------------
# Answer question
# -----------------------
def answer_question(question, vectorstore):
    # 1. Retrieve top 3 similar chunks from the FAISS index
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    # 2. Construct the prompt with the retrieved context
    prompt = f"Context from research documents:\n{context}\n\nQuestion: {question}"

    # 3. Generate response using the correct google-genai SDK syntax
    # Note: 'chat-bison' is deprecated; 'gemini-2.0-flash' or 'gemini-1.5-flash' is recommended
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config={
            "system_instruction": "You are a helpful research assistant. Use the provided context to answer questions accurately. If the answer isn't in the context, say you don't know.",
            "temperature": 0.2,
            "max_output_tokens": 800
        },
        contents=prompt
    )

    # 4. Return the text content from the response object
    return response.text