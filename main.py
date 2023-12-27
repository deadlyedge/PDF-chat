import os
import streamlit as st
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from qdrant_client import QdrantClient

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.qdrant import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
from utils import get_pdf_text, get_text_chunks
from dotenv import load_dotenv

load_dotenv()

# choose embedding model
embeddings = OpenAIEmbeddings()


def make_vectorstore(pdf_docs, collection_name):
    """
    make vector store from pdf docs
    """
    text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(text)

    vectors = Qdrant.from_texts(
        url=os.getenv("QDRANT_URL"),
        collection_name=collection_name,
        texts=text_chunks,
        embedding=embeddings,
        prefer_grpc=True,  # has to be, or creating will cause time out.
        force_recreate=True,
    )
    print("from create:", vectors)

    return


def get_db_collections() -> dict:
    """get collections name and vector count"""
    client = QdrantClient(os.getenv("QDRANT_URL"))

    collections_list = [
        collection["name"]
        for collection in client.get_collections().model_dump()["collections"]
    ]

    info_list = [
        client.get_collection(collection_name=collection).model_dump()["vectors_count"]
        for collection in collections_list
    ]

    return {"collection_name": collections_list, "caption": info_list}


def load_conversation_chain(collection):
    """
    load vector store from qdrant
    and then initial a conversation chain
    """
    # choose model (gpt-4 is better but expensive)
    llm = ChatOpenAI(model="gpt-4")

    # load vector store from qdrant
    client = QdrantClient(os.getenv("QDRANT_URL"))
    vectorstore = Qdrant(
        client=client, collection_name=collection, embeddings=embeddings
    )

    # load chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # make chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question):
    """
    use conversation chain to handle user input
    and update chat history in session state.
    """
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    print(st.session_state.chat_history)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # st.write(
            #     user_template.replace("{{MSG}}", message.content),
            #     unsafe_allow_html=True,
            # )
            with st.chat_message('user'):
                st.markdown(message.content)
        else:
            # st.write(
            #     bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            # )
            with st.chat_message('assistant'):
                st.markdown(message.content)


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    # st.write(css, unsafe_allow_html=True)

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
                options=db_list["collection_name"],
                captions=[f"vectors:{n}" for n in db_list["caption"]],
                horizontal=True,
            )

            if st.button("选择"):
                with st.spinner("读取中..."):
                    # create conversation chain
                    st.session_state.collection = db_selection
                    st.session_state.conversation = load_conversation_chain(
                        db_selection
                    )

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
                        # create vector store
                        make_vectorstore(pdf_docs, collection_name)
                        st.rerun()

        st.write("[qdrant database UI](http://192.168.50.16:6333/dashboard)")

    st.header(
        f"Chat with {st.session_state.collection if st.session_state.collection else 'your PDFs'} :books:"
    )
    if st.session_state.collection:
        if user_question := st.chat_input("Ask a question about your documents:"):
            handle_userinput(user_question)
    else:
        st.write("上传或者选择资料库")


if __name__ == "__main__":
    main()
