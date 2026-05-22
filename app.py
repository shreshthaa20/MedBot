from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import threading

# 1. Load the secret API key
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 2. Setup Flask App
app = Flask(__name__)

rag_chain = None
rag_lock = threading.Lock()

# 5. The LangChain Logic: How to format the answer
import re

def generate_answer(inputs):
    context = inputs.get("context", "")
    question = inputs.get("input", "")
    
    if not context.strip():
        return "Sorry, I could not find relevant information in the medical book for your question."
    
    # Clean up the context
    # Remove page numbers and formatting artifacts
    cleaned = re.sub(r'GALE ENCYCLOPEDIA OF MEDICINE \d+', '', context)
    cleaned = re.sub(r'GEM - \d+ to \d+ - A \d+/\d+/\d+ \d+:\d+ [AP]M Page \d+', '', cleaned)
    cleaned = re.sub(r'Page \d+', '', cleaned)
    
    # Remove excessive whitespace and newlines
    cleaned = re.sub(r'\n\n+', '\n\n', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned)
    cleaned = cleaned.strip()
    
    # Remove duplicate paragraphs
    paragraphs = cleaned.split('\n\n')
    seen = set()
    unique_paragraphs = []
    for para in paragraphs:
        para_clean = para.strip()
        if para_clean and para_clean not in seen:
            seen.add(para_clean)
            unique_paragraphs.append(para_clean)
    
    cleaned = '\n\n'.join(unique_paragraphs)
    
    return (
        f"Based on the Medical Book for your question '{question}':\n\n"
        f"{cleaned}\n\n"
        f"(Note: For professional advice, please consult a healthcare provider.)"
    )
     

def get_rag_chain():
    global rag_chain

    if rag_chain is not None:
        return rag_chain

    with rag_lock:
        if rag_chain is not None:
            return rag_chain

        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY must be set in the environment.")

        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from langchain_pinecone import PineconeVectorStore
        from src.helper import download_hugging_face_embeddings

        print("Loading Database Connection...")
        embeddings = download_hugging_face_embeddings()
        docsearch = PineconeVectorStore.from_existing_index(
            index_name="medicalbot",
            embedding=embeddings
        )
        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | RunnableLambda(generate_answer)
        )
        return rag_chain

# 6. Web Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})

@app.route("/get", methods=["POST"])
def chat():
    # Grab the user's message from the website
    msg = request.form.get("msg", "")
    if not msg:
        return jsonify({"error": "No message provided"}), 400
        
    print(f"User asking: {msg}")
    
    # Send the message through our LangChain pipeline
    response = get_rag_chain().invoke(msg)
    return str(response)

if __name__ == '__main__':
    port = int(os.getenv("PORT", "8080"))
    print(f"Starting Medical Bot Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
