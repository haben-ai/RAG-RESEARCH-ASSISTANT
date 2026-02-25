from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from vector_store import load_vector_store
from prompts import PROMPT_TEMPLATE

def answer_question(question):
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=Ollama(model="mistral"),
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    result = qa_chain({"query": question})

    return {
        "answer": result["result"],
        "sources_used": len(result["source_documents"])
    }