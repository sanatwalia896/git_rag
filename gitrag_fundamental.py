## this  code is for checking the working of teh RAG based github  context llm modeling

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

llm = ChatGroq(model="llama-3.1-8b-instant")
print(llm)


def process_with_gitingest(github_url):
    # or from URL
    summary, tree, content = ingest(github_url)
    return summary, tree, content


embeddings = OllamaEmbeddings(model="granite-embedding:30m")
github_url = "https://github.com/sanatwalia896/SQL_LLAMA"
summary, tree, content = process_with_gitingest(github_url)
docs = [
    Document(page_content=tree, metadata={"type": "structure"}),
    Document(page_content=summary, metadata={"type": "summary"}),
    Document(page_content=content, metadata={"type": "content"}),
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
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

document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
query = "Explain the app.py in detail to me  ?"
response = retrieval_chain.invoke({"input": query})
print(response["context"])
print(
    "\n \n ------------------------------------------------------------------------------"
)
print(response["answer"])
