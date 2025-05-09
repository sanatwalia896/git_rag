import os
import uuid
import gc
import streamlit as st
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from gitingest import ingest

# Load environment
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="Git LLAMA", page_icon="🦥", layout="wide")

# Initialize LLM and Session State
llm = ChatGroq(model="llama-3.1-8b-instant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.retrieval_chain = None
    st.session_state.repo_data = {}
    st.session_state.current_repo = None


def reset_state():
    st.session_state.chat_history.clear()
    st.session_state.retrieval_chain = None
    gc.collect()


def create_vector_store(docs):
    """Create and return an in-memory FAISS vector store from the documents."""
    embedding_model = OllamaEmbeddings(model="all-minilm:33m")
    return FAISS.from_documents(docs, embedding_model)


def create_retriever_chain_from_store(vector_store):
    """Create a retrieval chain using FAISS retriever and LLM document chain."""
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
        You are a code analyst bot. Use the following context to answer the question:

        Repository Info:
        {context}

        Chat History:
        {chat_history}

        Question:
        {input}

        Answer:
        """
    )

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever, document_chain)


def process_repository(url):
    """Ingest and prepare the repository, build retriever chain."""
    summary, tree, content = ingest(url)
    st.session_state.repo_data = {"summary": summary, "tree": tree, "content": content}

    docs = [
        Document(page_content=tree, metadata={"type": "structure"}),
        Document(page_content=summary, metadata={"type": "summary"}),
        Document(page_content=content, metadata={"type": "content"}),
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    vector_store = create_vector_store(split_docs)
    st.session_state.retrieval_chain = create_retriever_chain_from_store(vector_store)
    return True


# Sidebar
with st.sidebar:
    st.header("GitHub Repo Input")
    repo_url = st.text_input("Enter GitHub Repository URL")
    col1, col2 = st.columns(2)
    if col1.button("Load Repo") and repo_url:
        try:
            repo_name = repo_url.strip().split("/")[-1]
            if process_repository(repo_url):
                st.session_state.current_repo = repo_name
                st.success(f"Loaded {repo_name} successfully!")
        except Exception as e:
            st.error(f"Failed to load repository: {e}")
    if col2.button("Reset Chat"):
        reset_state()

# Main UI
st.title("Git LLAMA 🦥 - GitHub Code Chatbot")

if st.session_state.current_repo:
    st.markdown(f"### Current Repo: `{st.session_state.current_repo}`")

    with st.expander("📂 Repository Structure", expanded=False):
        st.code(st.session_state.repo_data.get("tree", ""))

    for msg in st.session_state.chat_history:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        st.markdown(f"**{role}:** {msg['content']}")

    user_query = st.text_input("Ask something about the code")
    if st.button("Submit Query") and user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        try:
            result = st.session_state.retrieval_chain.invoke(
                {
                    "input": user_query,
                    "chat_history": str(st.session_state.chat_history),
                }
            )
            answer = result.get("answer", "Sorry, no answer available.")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()
        except Exception as e:
            st.error(f"Failed to get response: {e}")
else:
    st.info("📥 Load a GitHub repository from the sidebar to begin.")
