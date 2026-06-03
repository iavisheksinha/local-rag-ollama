import fitz
from langchain_community.llms import Ollama

# Load PDF
doc = fitz.open("data/India Post.pdf")

# Extract text
text = ""

for page in doc:
    text += page.get_text()

# Load local AI model
llm = Ollama(model="llama3")

# User question
question = "Summarize this document simply."

# Create prompt
prompt = f"""
Here is a document:

{text[:5000]}

Answer this question:
{question}
"""

# Ask AI
response = llm.invoke(prompt)

# Print response
print(response)