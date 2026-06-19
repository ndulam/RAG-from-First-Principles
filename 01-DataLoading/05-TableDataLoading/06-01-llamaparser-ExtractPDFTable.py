from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_parse import LlamaParse
import time
from dotenv import load_dotenv

# Load environment variables (make sure an OpenAI API key is set)
load_dotenv()

# Set up the base models
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-3.5-turbo-0125")

Settings.llm = llm
Settings.embed_model = embed_model

# Define the PDF path
pdf_path = "90-Data/ComplexPDF/billionaires_page-1-5.pdf"

# Record the start time
start_time = time.time()

# Use LlamaParse to parse the PDF
documents = LlamaParse(result_type="markdown").load_data(pdf_path)

# Record the end time
end_time = time.time()
print(f"PDF parsing took: {end_time - start_time:.2f} seconds")

# Print the parsed result
print("\nParsed document content:")
for i, doc in enumerate(documents, 1):
    print(f"\nDocument {i} content:")
    print(doc.text)
