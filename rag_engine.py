from langchain_ollama import OllamaLLM
from vector_store import load_vector_store

MODEL_NAME = "phi3:mini"  # lightweight and fast


def answer_question(question: str):
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an academic research assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    llm = OllamaLLM(model=MODEL_NAME)

    response = llm.invoke(prompt)

    return {
        "answer": response,
        "sources_used": len(docs)
    }