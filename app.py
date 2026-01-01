import os
import time
from dotenv import load_dotenv
import streamlit as st

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from helper import load_pdf, get_chunks, get_vector_store

load_dotenv()

st.set_page_config(page_title="PDF Q&A Bot")
st.title("Ask Questions from your PDF 📄🤖")

pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf:
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf.read())

    text = load_pdf("uploaded.pdf")
    chunks = get_chunks(text)

    # ⚡ Cache vectorstore to avoid recomputing embeddings
    @st.cache_resource
    def build_vectorstore(chunks):
        return get_vector_store(chunks)

    vectorstore = build_vectorstore(chunks)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        llm = OllamaLLM(model="gemma3:latest", temperature=0)

        # ⚡ Faster prompt
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Answer the question using ONLY the context.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        )

        # ⚡ Faster retriever (fetch fewer docs)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # ⚡ Fast RAG chain
        rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )

        start_time = time.time()
        answer = rag_chain.invoke(query)
        end_time = time.time()

        st.write("Answer:", answer)
        st.write("Time taken:", round(end_time - start_time, 2), "seconds")
   