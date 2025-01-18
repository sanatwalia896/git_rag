import os

import gc
import tempfile
import uuid
import pandas as pd

from gitingest import ingest
import tempfile
import uuid
import langchain
from langchain_community.document_loaders import DirectoryLoader
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.title("GIT LLAMA 🦙 ")

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def process_with_gitingets(github_url):
    # or from URL
    summary, tree, content = ingest(github_url)
    return summary, tree, content


embeddings = OllamaEmbeddings(model="granite-embedding:30m")
with st.sidebar:
    st.header("Add Your github repository")
    github_url = st.text_input("Entr Github Repository URL", placeholder="GitHub URL")
    load_repo = st.button("Load Repository")
    if github_url and load_repo:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_name = github_url.split("/")[-1]
            file_key = f"{session_id}-{repo_name}"

            if file_key not in st.session_state.get("file_cache", {}):
                summary, tree, content = process_with_gitingets(github_url)

                content_path = os.path.join(temp_dir, f"{repo_name}_content.md")
                with open(content_path, "w", encoding="utf-8") as f:
                    f.write(content)
                loader = DirectoryLoader(path=temp_dir)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                documents = text_splitter.split_documents(docs)
