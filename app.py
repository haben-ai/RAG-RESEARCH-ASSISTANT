from flask import Flask, render_template, request, jsonify
from rag_engine import build_vector_store, load_vector_store, answer_question
import requests
from PyPDF2 import PdfReader

app = Flask(__name__)

VECTOR_STORE_PATH = "vectorstore"

# Load vector store if exists
try:
    vectorstore = load_vector_store()
except:
    vectorstore = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ingest", methods=["POST"])
def ingest():
    global vectorstore
    data = request.get_json()
    pdf_url = data.get("pdf_url")
    
    if not pdf_url:
        return jsonify({"status": "error", "message": "PDF URL missing"}), 400
    
    try:
        # Download PDF
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # Read PDF text
        reader = PdfReader(response.content)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

        # Build vector store
        num_docs = build_vector_store(text)
        vectorstore = load_vector_store()
        
        return jsonify({"status": "success", "chunks_indexed": num_docs})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    global vectorstore
    data = request.get_json()
    question = data.get("question")
    
    if not question:
        return jsonify({"status": "error", "message": "Question missing"}), 400
    if not vectorstore:
        return jsonify({"status": "error", "message": "No vector store found. Please ingest a PDF first."}), 400
    
    try:
        answer = answer_question(question, vectorstore)
        return jsonify({"status": "success", "answer": answer})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)