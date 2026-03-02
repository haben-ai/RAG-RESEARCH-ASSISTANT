RAG Research Assistant



A Retrieval-Augmented Generation (RAG) based research assistant built using:

Flask (Backend API)

LangChain

FAISS Vector Store

HuggingFace Embeddings

Google Gemini API (google-genai SDK)

Optional: Streamlit Frontend

This project allows users to ask questions about research documents. It retrieves the most relevant document chunks and generates context-aware answers using Gemini.

🚀 Features

Document chunking using RecursiveCharacterTextSplitter

Local embeddings using sentence-transformers/all-MiniLM-L6-v2

Vector search powered by FAISS

Context-aware question answering using Gemini

REST API endpoint (/ask)

Supports retry handling and API quota management

Modular RAG architecture

🧠 Architecture

User Question
↓
FAISS Similarity Search (Top K Chunks)
↓
Construct Prompt with Context
↓
Gemini generate_content()
↓
Return Context-Aware Answer

📁 Project Structure
RAG-RESEARCH-ASSISTANT/
│
├── app.py                 # Flask backend
├── rag_engine.py          # RAG logic
├── config.py              # Configurations (VECTOR_STORE_PATH)
├── streamlit_app.py       # Optional frontend
├── vectorstore/           # Saved FAISS index
├── .env                   # Environment variables (API Key)
└── README.md
![](https://github.com/haben-ai/RAG-RESEARCH-ASSISTANT/blob/main/RAG%202026-03-02%20174459.png)
⚙️ Installation
1️⃣ Clone the Repository
git clone 
cd RAG-RESEARCH-ASSISTANT
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
3️⃣ Install Dependencies
pip install -r requirements.txt

If you don't have a requirements file, install:

pip install flask langchain langchain-community langchain-text-splitters langchain-huggingface faiss-cpu sentence-transformers google-genai python-dotenv
🔑 Setting Up Gemini API Key

This project uses the Google Gemini API via google-genai.

Recommended: Use .env file

Create a .env file in the root directory:

GENAI_API_KEY=your_actual_api_key_here

Ensure your rag_engine.py loads it:

from dotenv import load_dotenv
load_dotenv()

⚠️ Never commit your API key to GitHub.

▶️ Running the Project
Start Flask Backend
python app.py

Server will run at:

http://127.0.0.1:5000
Example API Request

POST to /ask:

{
  "question": "What problem does the paper solve?"
}
Optional: Run Streamlit Frontend
streamlit run streamlit_app.py
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