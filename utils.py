from PyPDF2 import PdfReader
from icecream import ic
from langchain.text_splitter import CharacterTextSplitter


def is_contains_chinese(strs):
    for _char in strs:
        if "\u4e00" <= _char <= "\u9fa5":
            return True
    return False


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += (
                "".join(page.extract_text().split())
                if is_contains_chinese(text)
                else page.extract_text()
            )
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="。" if is_contains_chinese(text) else "\n",
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    ic(len(chunks))
    return chunks

