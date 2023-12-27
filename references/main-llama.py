import os
import qdrant_client
import streamlit as st
import requests

import google.generativeai as genai

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from references.htmlTemplates import css, bot_template, user_template

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from llama_index import StorageContext, VectorStoreIndex
from llama_index.service_context import ServiceContext
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.schema import Document
from llama_index.retrievers import VectorIndexAutoRetriever


load_dotenv()


def get_pdf_text(pdf_docs):
    """get text from pdf file
    add join() and split() to remove extra spaces from chinese texts
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += "".join(page.extract_text().split())
    return text


def build_index(text):
    text_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=50)
    embed_model = GeminiEmbedding()
    service_context = ServiceContext.from_defaults(
        text_splitter=text_splitter, embed_model=embed_model
    )
    document = [Document(text=text)]
    client = qdrant_client.QdrantClient("192.168.50.16:6333")
    vector_store = QdrantVectorStore(client=client, collection_name="zhuangzi")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=document,
        service_context=service_context,
        storage_context=storage_context,
    )
    print(index)
    return index


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # build index
                index = build_index(raw_text)

                # create conversation chain
                st.session_state.conversation = index.as_retriever()


if __name__ == "__main__":
    main()
