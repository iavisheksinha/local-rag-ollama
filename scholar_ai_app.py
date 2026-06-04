import streamlit as st
import pymupdf as fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# Page title
st.title("ScholarAI")

# Upload multiple PDFs
uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    # Extract text from all PDFs
    all_text = ""

    for uploaded_file in uploaded_files:

        doc = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )

        for page in doc:
            all_text += page.get_text() + "\n"

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_text(all_text)

    st.write(f"Total PDFs uploaded: {len(uploaded_files)}")
    st.write(f"Total chunks created: {len(chunks)}")

    # Create embeddings and vector DB
    with st.spinner("Processing PDFs and creating embeddings..."):

        embedding = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        vectorstore = Chroma.from_texts(
            chunks,
            embedding=embedding
        )

    st.success("PDFs processed successfully!")

    # Ask question
    question = st.text_input(
        "Ask a question about the uploaded PDFs"
    )

    if question:

        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(
            question,
            k=5
        )

        # Build context
        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        # Debug view
        with st.expander("Retrieved Context"):
            st.write(context)

        # Load LLM
        llm = OllamaLLM(
            model="llama3"
        )

        # Prompt
        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}
"""

        # Generate response
        with st.spinner("Generating answer..."):
            response = llm.invoke(prompt)

        # Display answer
        st.subheader("Answer")
        st.write(response)