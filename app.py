from flask import Flask, request, jsonify
from pdf_utils import download_pdf, extract_text_from_pdf
from vector_store import build_vector_store
from rag_engine import answer_question
import os

app = Flask(__name__)

@app.route("/ingest", methods=["POST"])
def ingest():
    data = request.json
    pdf_url = data.get("pdf_url")

    if not pdf_url:
        return jsonify({"error": "PDF URL required"}), 400

    try:
        file_path = download_pdf(pdf_url)
        text = extract_text_from_pdf(file_path)
        chunks = build_vector_store(text)

        return jsonify({
            "status": "success",
            "chunks_indexed": chunks
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question required"}), 400

    try:
        response = answer_question(question)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)