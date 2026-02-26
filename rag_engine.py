from google import genai
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from config import VECTOR_STORE_PATH
import os

# Initialize Gemini client (reads API key from GENAI_API_KEY environment variable)
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
    # Retrieve relevant documents
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(question)
    
    # Combine content into one prompt
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Call Gemini
    response = client.responses.create(
        model="models/gemini-2.5-flash",  # use a valid model from your list
        input=prompt
    )

    # Extract text output
    answer = ""
    for item in response.output:
        for content in item.content:
            if content.type == "output_text":
                answer += content.text

    return answer.strip()