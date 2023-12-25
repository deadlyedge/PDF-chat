import os
import streamlit as st
import requests
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from qdrant_client import QdrantClient

# from qdrant_client.http.models import VectorParams, Distance
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.qdrant import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),  # type: ignore
    task_type="retrieval_document",
)

db_client = QdrantClient(
    url="192.168.50.16:6333",
    # api_key=os.getenv('QDRANT_API_KEY')
)
doc_store = Qdrant(
    client=db_client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME") or "test",
    embeddings=embeddings,
)

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()  # " ".join(page.extract_text().split())
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=50, length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(len(chunks))
    return chunks


# def recreate_collection(client):
#     vectors_config = VectorParams(
#         size=768,
#         distance=Distance.COSINE,
#     )

#     client.recreate_collection(
#         collection_name=os.getenv("QDRANT_COLLECTION_NAME") or '',
#         vectors_config=vectors_config,
#     )


def make_vectorstore(text_chunks, collection_name):
    response = requests.get(
        f'{os.getenv("QDRANT_HOST")}/collections/{os.getenv('QDRANT_COLLECTION_NAME')}',
        timeout=3000,
    )

    print(response.json())

    if response.status_code != 200 or not response.json()["result"]["vectors_count"]:
        vector_store = doc_store.from_texts(
            texts=text_chunks,
            url="192.168.50.16:6333",
            collection_name=collection_name or "test",
            # api_key=os.getenv("QDRANT_API_KEY"),
            embedding=embeddings,
            # force_recreate=True
        )
        return vector_store
    else:
        return doc_store


def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", convert_system_message_to_human=True
    )  # type: ignore
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    # load vector store from qdrant

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


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


def get_db_collections() -> list:
    return [
        {"db_name": "11111", "caption": "11-11-1111"},
        {"db_name": "2222", "caption": "22-2-2222"},
        {"db_name": "33333", "caption": "3-3-3333"},
    ]


def load_db_collection(collection):
    print(collection)
    return doc_store


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "collection" not in st.session_state:
        st.session_state.collection = None

    with st.sidebar:
        with st.container(border=True):
            # st.subheader("你的资料库")

            db_list = get_db_collections()
            db_selection = st.radio(
                label="选择一个资料库",
                options=[db["db_name"] for db in db_list],
                captions=[db["caption"] for db in db_list],
                horizontal=True,
            )
            print(db_selection)

            if st.button("选择"):
                with st.spinner("读取中..."):
                    # create conversation chain
                    st.session_state.collection = db_selection
                    vectorstore = load_db_collection(db_selection)
                    st.session_state.conversation = get_conversation_chain(vectorstore)

        with st.container(border=True):
            # st.subheader("Your documents")
            collection_name = st.text_input(
                label="资料库名称",
                help="指定数据库collection name",
                placeholder="research_1",
            )
            pdf_docs = st.file_uploader(
                "上传PDF文件",
                accept_multiple_files=True,
                type=["pdf"],
            )

            if collection_name and pdf_docs:
                if st.button("处理"):
                    with st.spinner("处理中..."):
                        # get pdf text
                        raw_text = get_pdf_text(pdf_docs)

                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # create vector store
                        make_vectorstore(text_chunks, collection_name)

    st.header(
        f"Chat with {st.session_state.collection if st.session_state.collection else 'your PDFs'} :books:"
    )
    if st.session_state.collection:
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)
    else:
        st.write("上传或者选择资料库")


if __name__ == "__main__":
    main()
