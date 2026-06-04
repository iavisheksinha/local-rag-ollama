import streamlit as st
import pymupdf as fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# Page title
st.title("ScholarAI")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_chunks = []
    all_metadatas = []

    # Process PDFs
    for uploaded_file in uploaded_files:

        pdf_name = uploaded_file.name

        doc = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )

        text = ""

        for page in doc:
            text += page.get_text()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)

        for chunk in chunks:
            all_chunks.append(chunk)

            all_metadatas.append(
                {
                    "source": pdf_name
                }
            )

    st.write(f"Total PDFs uploaded: {len(uploaded_files)}")
    st.write(f"Total chunks created: {len(all_chunks)}")

    # Create Vector Store
    with st.spinner("Processing PDFs and creating embeddings..."):

        embedding = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        vectorstore = Chroma.from_texts(
            texts=all_chunks,
            embedding=embedding,
            metadatas=all_metadatas
        )

    st.success("PDFs processed successfully!")

    # Question box
    question = st.text_input(
        "Ask a question about the uploaded PDFs"
    )

    # Buttons
    generate_mcq = st.button(
        "Generate 5 MCQs"
    )

    summarize_pdf = st.button(
        "Generate Summary"
    )

    # ---------------------------
    # QUESTION ANSWERING
    # ---------------------------

    if question:

        docs = vectorstore.similarity_search(
            question,
            k=5
        )

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        sources = set()

        for doc in docs:
            if "source" in doc.metadata:
                sources.add(doc.metadata["source"])

        with st.expander("Retrieved Context"):
            st.write(context)

        llm = OllamaLLM(
            model="llama3"
        )

        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}
"""

        with st.spinner("Generating answer..."):
            response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response)

        st.subheader("Sources")

        for source in sorted(sources):
            st.write(f"📄 {source}")

    # ---------------------------
    # MCQ GENERATION
    # ---------------------------

    if generate_mcq:

        docs = vectorstore.similarity_search(
            "Generate MCQs from this document",
            k=5
        )

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        llm = OllamaLLM(
            model="llama3"
        )

        mcq_prompt = f"""
You are an expert teacher.

Using the context below, generate exactly 5 multiple-choice questions.

Rules:
- Each question must have 4 options
- Label options A, B, C, D
- Show the correct answer after each question
- Use only information from the provided context

Context:
{context}
"""

        with st.spinner("Generating 5 MCQs..."):
            mcqs = llm.invoke(mcq_prompt)

        st.subheader("Generated MCQs")
        st.write(mcqs)

    # ---------------------------
    # PDF SUMMARY
    # ---------------------------

    if summarize_pdf:

        docs = vectorstore.similarity_search(
            "Summarize this document",
            k=10
        )

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        llm = OllamaLLM(
            model="llama3"
        )

        summary_prompt = f"""
You are an expert academic assistant.

Create a structured summary of the provided content.

Use the following format:

1. Overview
2. Key Concepts
3. Important Points
4. Conclusion

Keep the summary concise and easy to understand.

Context:
{context}
"""

        with st.spinner("Generating summary..."):
            summary = llm.invoke(summary_prompt)

        st.subheader("Document Summary")
        st.write(summary)