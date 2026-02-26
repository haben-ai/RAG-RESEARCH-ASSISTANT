from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai
from config import VECTOR_STORE_PATH
import os

# Initialize Gemini client
client = genai.Client()

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


def answer_question(question, vectorstore):
    # Retrieve top relevant docs
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    # Use Gemini API to generate answer
    response = client.responses.create(
        model="gemini-1.5-t",  # Choose a valid model
        input=f"Use the following context to answer the question:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    )

    # response.output_text contains the generated text
    return response.output_text