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
