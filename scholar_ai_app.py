import streamlit as st
import pymupdf as fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM 

# Page title
st.title("ScholarAI")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Read PDF
    doc = fitz.open("temp.pdf")

    # Extract text
    text = ""

    for page in doc:
        text += page.get_text()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    st.write(f"Total chunks created: {len(chunks)}")

    # Process embeddings + vector DB
    with st.spinner("Processing PDF and creating embeddings..."):

        # Embedding model
        embedding = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        # Create vector database
        vectorstore = Chroma.from_texts(
            chunks,
            embedding=embedding
        )

    st.success("PDF processed successfully!")

    # User question
    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(
            question,
            k=5
        )

        # Combine retrieved chunks
        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        # Debug retrieved context
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