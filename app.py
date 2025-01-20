import os
from langchain.schema import Document
import gc

import uuid
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain

from gitingest import ingest


from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
st.title("GIT LLAMA 🦙 ")
llm = ChatGroq(model="llama-3.1-8b-instant")

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []
    st.session_state.context = None

session_id = st.session_state.id
client = None


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def process_with_gitingest(github_url):
    # or from URL
    summary, tree, content = ingest(github_url)
    return summary, tree, content


embeddings = OllamaEmbeddings(model="granite-embedding:30m")
with st.sidebar:
    st.header("Add Your github repository")
    github_url = st.text_input("Entr Github Repository URL", placeholder="GitHub URL")
    load_repo = st.button("Load Repository")

    if github_url and load_repo:
        repo_name = github_url.split("/")[-1]
        file_key = f"{session_id}-{repo_name}"

        if file_key not in st.session_state.get("file_cache", {}):
            summary, tree, content = process_with_gitingest(github_url)

            # Create documents from string content instead of loading from files
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
                vector_store = FAISS.from_documents(
                    documents=documents, embedding=embeddings
                )
                retriever = vector_store.as_retriever()

                prompt = ChatPromptTemplate.from_template(
                    """
                    You are an AI assistant specialized in analyzing GitHub repositories.

                    Repository structure:

                    Context information from the repository:
                    {context}
                    ---------------------

                    Given the repository structure and context above, provide a clear and precise answer to the query.
                    Focus on the repository's content, code structure, and implementation details.
                    If the information is not available in the context, respond with 'I don't have enough information about that aspect of the repository.'

                    Query: {input}
                    Answer: """,
                )

                query = st.text_input("Enter your query here")
                output_parser = StrOutputParser()
                document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                # Add a button to submit the query
                if st.button("Submit Query"):
                    if query:
                        result = retrieval_chain.invoke(
                            {  # Optionally, provide context information
                                "query": query,
                                "context": "\n".join(
                                    [doc.page_content for doc in documents]
                                ),  # Add context
                            }
                        )
                        st.write("Answer:", result["answer"])

                    else:
                        st.error("Please enter a query before submitting.")

                    # Cache loaded repo content
                    st.session_state.file_cache[file_key] = (summary, tree, content)

    st.button("Reset Chat", on_click=reset_chat)
