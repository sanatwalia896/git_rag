# import os
# import uuid
# import gc
# import time
# import streamlit as st
# from langchain.schema import Document
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.retrieval import create_retrieval_chain
# from gitingest import ingest
# from langchain_groq import ChatGroq
# from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from dotenv import load_dotenv
# from langchain.chains.combine_documents import create_stuff_documents_chain

# # Load environment variables
# load_dotenv()

# # Configure Streamlit page
# st.set_page_config(page_title="Git LLAMA", page_icon="🦙", layout="wide")

# # Custom CSS for better UI
# st.markdown(
#     """
#     <style>
#     .chat-message { padding: 1rem; margin: 1rem 0; border-radius: 0.5rem; }
#     .file-structure { font-family: monospace; white-space: pre-wrap; }
#     </style>
# """,
#     unsafe_allow_html=True,
# )

# # Initialize the language model
# llm = ChatGroq(model="llama-3.1-8b-instant")

# # Enhanced session state management
# if "id" not in st.session_state:
#     st.session_state.id = uuid.uuid4()
#     st.session_state.file_cache = {}
#     st.session_state.chat_history = {}  # Now a dict to store history per repo
#     st.session_state.retrieval_chains = {}  # Store chains for each repo
#     st.session_state.current_repo = None
#     st.session_state.repo_structures = {}  # Store structures for each repo

# session_id = st.session_state.id

# def reset_chat(repo_name=None):
#     """Reset the chat session for a specific repository or all repositories."""
#     if repo_name:
#         if repo_name in st.session_state.chat_history:
#             st.session_state.chat_history[repo_name] = []
#     else:
#         st.session_state.chat_history = {}
#     gc.collect()

# def process_with_gitingest(github_url, chunk_size=1000):
#     """Process the GitHub repository using gitingest with error handling and chunking."""
#     try:
#         with st.spinner("Analyzing repository structure..."):
#             summary, tree, content = ingest(github_url)
#             return summary, tree, content
#     except Exception as e:
#         st.error(f"Error during repository ingestion: {str(e)}")
#         return None, None, None

# def create_retrieval_chain_from_docs(documents, repo_name):
#     """Create a retrieval chain from documents with enhanced error handling."""
#     try:
#         embeddings = OllamaEmbeddings(model="granite-embedding:30m")

#         with st.spinner("Creating vector store..."):
#             vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
#             retriever = vector_store.as_retriever(
#                 search_kwargs={"k": 4}  # Adjust number of retrieved documents
#             )

#         prompt_template = """
#         You are an AI assistant specialized in analyzing GitHub repositories.

#         Repository: {repo_name}

#         Context from repository:
#         {context}
#         ---------------------

#         Previous conversation:
#         {chat_history}
#         ---------------------

#         Given the repository structure, context, and our previous conversation, provide a clear and precise answer to the query.
#         Focus on the repository's content, code structure, and implementation details.
#         If referring to specific files or code sections, mention their paths.
#         If the information is not available in the context, say so clearly.

#         Query: {input}
#         Answer: """

#         prompt = ChatPromptTemplate.from_template(prompt_template)
#         document_chain = create_stuff_documents_chain(
#             llm=llm,
#             prompt=prompt,
#             document_variable_name="context",
#         )
#         return create_retrieval_chain(retriever, document_chain)
#     except Exception as e:
#         st.error(f"Error creating retrieval chain: {str(e)}")
#         return None

# def process_repository(github_url):
#     """Process a repository with enhanced error handling and progress tracking."""
#     try:
#         repo_name = github_url.split("/")[-1]
#         file_key = f"{session_id}-{repo_name}"

#         # Initialize chat history for new repository
#         if repo_name not in st.session_state.chat_history:
#             st.session_state.chat_history[repo_name] = []

#         progress_bar = st.progress(0)

#         # Process repository content
#         summary, tree, content = process_with_gitingest(github_url)
#         if not all([summary, tree, content]):
#             return False

#         progress_bar.progress(40)

#         # Create documents with chunking
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,  # Reduced chunk size for better processing
#             chunk_overlap=50,
#             length_function=len,
#         )

#         docs = [
#             Document(page_content=tree, metadata={"type": "structure", "repo": repo_name}),
#             Document(page_content=summary, metadata={"type": "summary", "repo": repo_name}),
#             Document(page_content=content, metadata={"type": "content", "repo": repo_name}),
#         ]

#         progress_bar.progress(60)

#         # Split documents with error handling
#         try:
#             documents = text_splitter.split_documents(docs)
#         except Exception as e:
#             st.error(f"Error splitting documents: {str(e)}")
#             return False

#         progress_bar.progress(80)

#         # Create retrieval chain
#         retrieval_chain = create_retrieval_chain_from_docs(documents, repo_name)
#         if not retrieval_chain:
#             return False

#         # Update session state
#         st.session_state.retrieval_chains[repo_name] = retrieval_chain
#         st.session_state.repo_structures[repo_name] = tree
#         st.session_state.file_cache[file_key] = (summary, tree, content)
#         st.session_state.current_repo = repo_name

#         progress_bar.progress(100)
