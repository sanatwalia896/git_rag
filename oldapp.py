import os
import uuid
import gc
import streamlit as st
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from gitingest import ingest
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Streamlit app title
st.title("GIT LLAMA 🦙 ")

# Initialize the language model
llm = ChatGroq(model="llama-3.1-8b-instant")

# Session state management
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []
    st.session_state.context = None
    st.session_state.query_submitted = False
    st.session_state.retrieval_chain = None

session_id = st.session_state.id


def reset_chat():
    """Reset the chat session."""
    st.session_state.messages = []
    st.session_state.context = None
    st.session_state.retrieval_chain = None
    gc.collect()


def process_with_gitingest(github_url):
    """Process the GitHub repository using gitingest."""
    summary, tree, content = ingest(github_url)
    return summary, tree, content


def create_retrieval_chain_from_docs(documents):
    """Create a retrieval chain from documents."""
    embeddings = OllamaEmbeddings(model="granite-embedding:30m")
    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    retriever = vector_store.as_retriever()

    prompt_template = """
    You are an AI assistant specialized in analyzing GitHub repositories.

    Repository structure:

    Context information from the repository:
    {context}
    ---------------------

    Given the repository structure and context above, provide a clear and precise answer to the query.
    Focus on the repository's content, code structure, and implementation details.
    If the information is not available in the context, respond with 'I don't have enough information about that aspect of the repository.'

    Query: {input}
    Answer: """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever, document_chain)


# Sidebar for repository loading
with st.sidebar:
    st.header("Add Your GitHub Repository")
    github_url = st.text_input("Enter GitHub Repository URL", placeholder="GitHub URL")
    load_repo = st.button("Load Repository")

    if github_url and load_repo:
        with st.spinner("Loading repository..."):
            repo_name = github_url.split("/")[-1]
            file_key = f"{session_id}-{repo_name}"

            if file_key not in st.session_state.file_cache:
                summary, tree, content = process_with_gitingest(github_url)

                # Create documents from string content
                docs = [
                    Document(page_content=tree, metadata={"type": "structure"}),
                    Document(page_content=summary, metadata={"type": "summary"}),
                    Document(page_content=content, metadata={"type": "content"}),
                ]

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                documents = text_splitter.split_documents(docs)

                if not documents:
                    st.error("Failed to process documents from repository content.")
                else:
                    # Create and store the retrieval chain
                    st.session_state.retrieval_chain = create_retrieval_chain_from_docs(
                        documents
                    )
                    st.session_state.file_cache[file_key] = (summary, tree, content)
                    st.success("Repository loaded successfully!")

    # Button to reset chat session
    st.button("Reset Chat", on_click=reset_chat)

# Main chat interface
st.header("Chat with your Repository")

# Only show query input if a repository is loaded
if st.session_state.retrieval_chain is not None:
    query = st.text_input("Enter your query about the repository")

    if st.button("Submit Query"):
        if query:
            with st.spinner("Processing query..."):
                try:
                    result = st.session_state.retrieval_chain.invoke({"input": query})
                    if "answer" in result:
                        st.write("Answer:", result["answer"])
                    else:
                        st.error("No answer returned from the model.")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a query before submitting.")
else:
    st.info("Please load a GitHub repository first using the sidebar.")
