<h2>RAG Research Assistant </h2>

![](https://github.com/haben-ai/RAG-RESEARCH-ASSISTANT/blob/main/rag.png)

# A Retrieval-Augmented Generation (RAG) based research assistant built using:


1. LangChain

2. FAISS Vector Store

3. HuggingFace Embeddings

4. Google Gemini API (google-genai SDK)

5. Streamlit Frontend

This project allows users to ask questions about research documents. It retrieves the most relevant document chunks and generates context-aware answers using Gemini.

# Features

1. Document chunking using RecursiveCharacterTextSplitter

2. Local embeddings using sentence-transformers/all-MiniLM-L6-v2

3. Vector search powered by FAISS

4. Context-aware question answering using Gemini


Modular RAG architecture

# 🧠 Architecture

 User Question
 ↓
 FAISS Similarity Search (Top K Chunks)
 ↓
 Construct Prompt with Context
 ↓
 Gemini generate_content()
 ↓
 Return Context-Aware Answer

# 📁 Project Structure
RAG-RESEARCH-ASSISTANT/

![](https://github.com/haben-ai/RAG-RESEARCH-ASSISTANT/blob/main/rag-dir.png)
```
⚙️ Installation
1️⃣ Clone the Repository
git clone 
cd RAG-RESEARCH-ASSISTANT
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate  # Windows
 source venv/bin/activate  # Mac/Linux
3️⃣ Install Dependencies
pip install -r requirements.txt```


![](https://github.com/haben-ai/RAG-RESEARCH-ASSISTANT/blob/main/RAG%202026-03-02%20174459.png)
```
pip install flask langchain langchain-community langchain-text-splitters langchain-huggingface faiss-cpu sentence-transformers google-genai python-dotenv ```
```
🔑 Setting Up Gemini API Key



GENAI_API_KEY=your_actual_api_key_here



from dotenv import load_dotenv
load_dotenv()



🧩 How RAG Works in This Project
1. Document Chunking

Uses:

RecursiveCharacterTextSplitter
2. Embeddings

Uses:

HuggingFaceEmbeddings
Model: sentence-transformers/all-MiniLM-L6-v2

This avoids API embedding costs.

3. Vector Search

Uses:

FAISS
4. LLM Generation

Uses:

client.models.generate_content()
Model: gemini-2.0-flash