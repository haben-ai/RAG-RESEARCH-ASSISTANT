from flask import Flask, request, jsonify
from rag_engine import build_vector_store, load_vector_store, answer_question

app = Flask(__name__)
vectorstore = None

@app.route("/ask", methods=["POST"])
def ask():
    global vectorstore
    data = request.get_json()
    paper_url = data.get("paper_url")
    question = data.get("question")

    if not vectorstore:
        # For now, dummy text from paper; in real use, download PDF and extract text
        text = "Insert extracted paper text here..."
        build_vector_store(text)
        vectorstore = load_vector_store()

    answer = answer_question(question, vectorstore)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)