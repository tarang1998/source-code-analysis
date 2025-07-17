import os
from typing_extensions import override
from flask import Flask, render_template, request, jsonify, session
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from src.helper import SUPPORTED_LANGUAGES, clone_rep, computing_embeddings, load_embeddings_status, load_repo, process_repo_documents, save_embeddings_status, valid_github_url, load_vectordb
from langchain_openai import ChatOpenAI




app = Flask(__name__)
load_dotenv(override=True)

OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
app.secret_key = "your-very-secret-key" 

repo_cache = {}


REPO_BASE = "test_repo"
DB_DIR = "db"


os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(REPO_BASE, exist_ok=True)

embeddings_status = load_embeddings_status()




@app.route("/", methods=["GET"])
def index():
    # analyzed = session.get("repo_url") in repo_cache
    return render_template("index.html", analyzed=False)


@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.form["github_url"].strip()
    if not valid_github_url(url):
        return render_template("index.html", error="Invalid GitHub URL."), 400

    repo_path = clone_rep(url)

    # Check persistent embeddings status
    embeddings_computed = embeddings_status.get(url, False)

    vectordb = None 

    if not embeddings_computed:

        documents = load_repo(repo_path=repo_path)

        processed_docs = process_repo_documents(documents)

        vectordb = computing_embeddings(processed_docs)

        embeddings_status[url] = True
        
        save_embeddings_status(embeddings_status)

    else:
        print("Embedding already computed")
        
        vectordb = load_vectordb()

    llm = ChatOpenAI()
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}),
        memory=memory
    )
       
    repo_cache[url] = {"qa_chain": qa_chain}

    session["repo_url"] = url

    return render_template("index.html", analyzed=True)



@app.route("/ask", methods=["POST"])
def ask():
    url = session.get("repo_url")
    if not url or url not in repo_cache:
        return jsonify({"answer": "Please analyze a repo first."})
    question = request.json["question"] if request.json and "question" in request.json else None
    if not question:
        return jsonify({"answer": "No question provided."})
    qa_chain = repo_cache[url]["qa_chain"]
    result = qa_chain({"question": question})
    return jsonify({"answer": result["answer"]})



if __name__ == "__main__":
    app.run(debug=True) 