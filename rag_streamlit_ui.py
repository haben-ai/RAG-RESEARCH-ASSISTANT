import streamlit as st
import requests

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="📚 AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

# -----------------------
# Custom Styling
# -----------------------
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
        }
        .stTextInput > div > div > input {
            background-color: #1e1e1e;
            color: white;
        }
        .stTextArea textarea {
            background-color: #1e1e1e;
            color: white;
        }
        .answer-box {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #333;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.title("📚 AI Research Assistant")
st.markdown("Ask questions about any research paper using Retrieval-Augmented Generation.")

st.divider()

# -----------------------
# Layout Columns
# -----------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔗 Paper URL")
    paper_url = st.text_input(
        "Enter link to research paper (PDF URL)",
        placeholder="https://arxiv.org/pdf/1706.03762.pdf"
    )

    st.subheader("❓ Your Question")
    user_question = st.text_area(
        "Ask something about the paper",
        placeholder="What is the main contribution of this paper?",
        height=150
    )

    ask_button = st.button("🚀 Ask RAG", use_container_width=True)

with col2:
    st.subheader("🤖 RAG Response")
    response_container = st.empty()

# -----------------------
# Backend Call
# -----------------------
if ask_button:
    if not paper_url or not user_question:
        st.warning("Please provide both a paper URL and a question.")
    else:
        with st.spinner("Processing paper and generating answer..."):
            try:
                res = requests.post(
                    "http://127.0.0.1:5000/ask",
                    json={
                        "paper_url": paper_url,
                        "question": user_question
                    }
                )

                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No answer returned.")

                    response_container.markdown(
                        f"""
                        <div class='answer-box'>
                        {answer}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.error(f"Backend Error: {res.text}")

            except Exception as e:
                st.error(f"Connection error: {e}")

st.divider()
st.caption("Built with Streamlit + Local RAG + Ollama")
