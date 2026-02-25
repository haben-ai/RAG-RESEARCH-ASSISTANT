from langchain_openai import ChatOpenAI
from vector_store import load_vector_store
from prompts import PROMPT_TEMPLATE

def answer_question(question):
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": len(docs)
    }