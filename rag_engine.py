import os
from google import genai
from vector_store import load_vector_store

# Create Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-1.5-flash"  # modern, fast, cheap


def answer_question(question: str):
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an academic research assistant.

Answer the question using ONLY the context below.
If the answer is not present, say "The paper does not mention this."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return {
        "answer": response.text,
        "sources_used": len(docs)
    }