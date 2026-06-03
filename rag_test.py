import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# Read PDF
doc = fitz.open("data/Int J Communication - 2024 - Sinha - Hybrid Whale Optimization‐Based Energy‐Efficient Lightweight Inte.pdf")

text = ""
for page in doc:
    text += page.get_text()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")

# Create embeddings
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Store in vector DB
vectorstore = Chroma.from_texts(
    chunks,
    embedding=embedding
)

# User question
question = "what is the value of inertia weight taken?"

# Retrieve relevant chunks
docs = vectorstore.similarity_search(question, k=3)

context = "\n".join([doc.page_content for doc in docs])
#added
for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i} ---\n")
    print(chunk)
# Load AI model
llm = Ollama(model="llama3")

# Prompt
prompt = f"""
Use this context to answer the question.

Context:
{context}

Question:
{question}
"""

# Generate answer
response = llm.invoke(prompt)

print(response)