import os
import uuid
import gc
import time
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

# Configure Streamlit page
st.set_page_config(page_title="Git LLAMA", page_icon="🦙", layout="wide")

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .chat-message { padding: 1rem; margin: 1rem 0; border-radius: 0.5rem; }
    

    .file-structure { font-family: monospace; white-space: pre-wrap; }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize the language model
llm = ChatGroq(model="llama-3.1-8b-instant")

# Enhanced session state management
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.chat_history = []
    st.session_state.context = None
    st.session_state.query_submitted = False
    st.session_state.retrieval_chain = None
    st.session_state.current_repo = None
    st.session_state.repo_structure = None

session_id = st.session_state.id


def reset_chat():
    """Reset the chat session while preserving repository data."""
    st.session_state.chat_history = []
    st.session_state.context = None
    gc.collect()


def process_with_gitingest(github_url):
    """Process the GitHub repository using gitingest with progress tracking."""
    with st.spinner("Analyzing repository structure..."):
        summary, tree, content = ingest(github_url)
    return summary, tree, content


def create_retrieval_chain_from_docs(documents):
    """Create a retrieval chain from documents with enhanced prompting."""
    embeddings = OllamaEmbeddings(model="granite-embedding:30m")

    with st.spinner("Creating vector store..."):
        vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
        retriever = vector_store.as_retriever()

    prompt_template = """
    You are an AI assistant specialized in analyzing GitHub repositories.

    Repository structure:
    {context}
    ---------------------

    Previous conversation:
    {chat_history}
    ---------------------

    Given the repository structure, context, and our previous conversation, provide a clear and precise answer to the query.
    Focus on the repository's content, code structure, and implementation details.
    If referring to specific files or code sections, mention their paths.
    If the information is not available in the context, say so clearly.

    Query: {input}
    Answer: """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever, document_chain)


def display_chat_history():
    """Display chat history with improved formatting."""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f"""
            <div class="chat-message user-message">
                <b>You:</b> {message["content"]}
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="chat-message assistant-message">
                <b>Assistant:</b> {message["content"]}
            </div>
            """,
                unsafe_allow_html=True,
            )


def display_repo_structure():
    """Display repository structure in a collapsible section."""
    if st.session_state.repo_structure:
        with st.expander("Repository Structure", expanded=False):
            st.markdown(
                '<div class="file-structure">'
                + st.session_state.repo_structure
                + "</div>",
                unsafe_allow_html=True,
            )


# Sidebar with enhanced repository management
with st.sidebar:
    st.header("Repository Management")
    github_url = st.text_input(
        "Enter GitHub Repository URL", placeholder="https://github.com/username/repo"
    )

    col1, col2 = st.columns(2)
    with col1:
        load_repo = st.button("Load Repository")
    with col2:
        st.button("Reset Chat", on_click=reset_chat)

    if st.session_state.current_repo:
        st.success(f"Current Repository: {st.session_state.current_repo}")

    if github_url and load_repo:
        progress_bar = st.progress(0)
        try:
            repo_name = github_url.split("/")[-1]
            file_key = f"{session_id}-{repo_name}"

            progress_bar.progress(20)

            if file_key not in st.session_state.file_cache:
                summary, tree, content = process_with_gitingest(github_url)
                progress_bar.progress(40)

                docs = [
                    Document(page_content=tree, metadata={"type": "structure"}),
                    Document(page_content=summary, metadata={"type": "summary"}),
                    Document(page_content=content, metadata={"type": "content"}),
                ]

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                progress_bar.progress(60)

                documents = text_splitter.split_documents(docs)
                progress_bar.progress(80)

                if documents:
                    st.session_state.retrieval_chain = create_retrieval_chain_from_docs(
                        documents
                    )
                    st.session_state.file_cache[file_key] = (summary, tree, content)
                    st.session_state.current_repo = repo_name
                    st.session_state.repo_structure = tree
                    progress_bar.progress(100)
                    st.success("Repository loaded successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to process repository content.")
            else:
                st.info("Loading cached repository...")
                summary, tree, content = st.session_state.file_cache[file_key]
                st.session_state.current_repo = repo_name
                st.session_state.repo_structure = tree
                progress_bar.progress(100)
                st.success("Repository loaded from cache!")

        except Exception as e:
            st.error(f"Error loading repository: {str(e)}")
        finally:
            progress_bar.empty()

# Main chat interface with enhanced features
st.title("GIT LLAMA 🦙")

# Display repository structure if available
display_repo_structure()

# Display chat history
display_chat_history()

# Query interface
if st.session_state.retrieval_chain is not None:
    query = st.text_input("Ask about the repository", key="query_input")

    if st.button("Submit Query", key="query_button"):
        if query:
            with st.spinner("Processing query..."):
                try:
                    # Add user message to chat history
                    st.session_state.chat_history.append(
                        {"role": "user", "content": query}
                    )

                    # Get response from model
                    result = st.session_state.retrieval_chain.invoke(
                        {
                            "input": query,
                            "chat_history": str(st.session_state.chat_history),
                        }
                    )

                    if "answer" in result:
                        # Add assistant response to chat history
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": result["answer"]}
                        )
                        st.rerun()
                    else:
                        st.error("No answer returned from the model.")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a query before submitting.")
else:
    st.info("Please load a GitHub repository first using the sidebar.")

# Footer with repository info
if st.session_state.current_repo:
    st.markdown("---")
    st.markdown(f"Currently analyzing: **{st.session_state.current_repo}**")
