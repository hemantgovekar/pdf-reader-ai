import os
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


def load_pdf(file_path):
    """Extract text from a PDF safely."""
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def get_chunks(text):
    """Split text into optimized chunks for embedding models."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,       # safe for embedding models like nomic-embed-text
        chunk_overlap=50,
        length_function=len
    )
    return splitter.split_text(text)


def get_vector_store(chunks):
    """Create a FAISS vectorstore using local Ollama embeddings."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore


# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.embeddings import OpenAIEmbeddings
# # from langchain.vectorstores import FAISS
# # from PyPDF2 import PdfReader
# # import os

# from langchain_classic.text_splitter import CharacterTextSplitter
# # from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings

# from langchain_classic.vectorstores import FAISS
# from PyPDF2 import PdfReader
# import os

# def load_pdf(file_path):
#     reader = PdfReader(file_path)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# def get_chunks(text):
#     splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     return splitter.split_text(text)

# # def get_vector_store(chunks):
# #     embeddings = OpenAIEmbeddings()
# #     vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
# #     return vectorstore

# def get_vector_store(chunks):
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
#     return vectorstore

