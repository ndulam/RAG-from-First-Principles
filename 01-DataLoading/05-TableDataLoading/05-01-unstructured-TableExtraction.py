"""
Use the unstructured library to extract tables from a PDF

[System dependencies]
Before running this script, install the following system dependencies:

1. Install poppler-utils (for PDF processing):
   sudo apt update
   sudo apt install -y poppler-utils

2. Install Tesseract OCR (for text recognition):
   sudo apt install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng

[Common error fixes]
- Error: "PDFInfoNotInstalledError: Unable to get page count. Is poppler installed and in PATH?"
  Fix: install poppler-utils

- Error: "TesseractNotFoundError: tesseract is not installed or it's not in your PATH"
  Fix: install tesseract-ocr

- Error: "No such file or directory" or a non-ASCII path showing up as garbled encoding
  Fix: make sure you run the script from the project root, or use an absolute path

[Verifying the install]
You can verify the install succeeded with:
- pdfinfo -v
- tesseract --version

[Notes for running in VSCode]
- Make sure the VSCode terminal's working directory is the project root
- If you hit path encoding issues, try running the script from the VSCode terminal
- It's recommended to set the Python interpreter in VSCode to the correct virtual environment

[Python dependencies]
91-Environment/requirements_llamaindex_Ubuntu-with-CPU.txt

"""

import os
import sys
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

# Make sure the working directory is correct
# Get the parent directory of the script's directory (the project root)
script_dir = Path(__file__).parent.parent.parent
if script_dir.exists():
    os.chdir(script_dir)
    print(f"Working directory set to: {os.getcwd()}")

# Import the relevant LlamaIndex modules
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables from the .env file
load_dotenv()

# Global settings
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

# Parse the PDF structure and extract text and tables
# Use a relative path, assuming we start from the project root
file_path = "90-Data/ComplexPDF/billionaires_page-1-5.pdf"

# Check whether the file exists
if not os.path.exists(file_path):
    print(f"Error: file not found - {file_path}")
    print(f"Current working directory: {os.getcwd()}")
    print("Please make sure:")
    print("1. You run the script from the project root")
    print("2. The PDF file path is correct")
    sys.exit(1)

print(f"Processing file: {file_path}")

elements = partition_pdf(
    file_path,
    strategy="hi_res",  # use the high-precision strategy
)  # parse the PDF document

# Build a mapping from element ID to element
element_map = {element.id: element for element in elements if hasattr(element, 'id')}

for element in elements:
    if element.category == "Table": # only print table data
        print("\nTable data:")
        print("Table metadata:", vars(element.metadata))  # use vars() to show all metadata attributes
        print("Table content:")
        print(element.text)  # print the table's text content

        # Get and print the parent node info
        parent_id = getattr(element.metadata, 'parent_id', None)
        if parent_id and parent_id in element_map:
            parent_element = element_map[parent_id]
            print("\nParent node info:")
            print(f"Type: {parent_element.category}")
            print(f"Content: {parent_element.text}")
            if hasattr(parent_element, 'metadata'):
                print(f"Parent node metadata: {vars(parent_element.metadata)}")  # also use vars() to show all metadata
        else:
            print(f"Parent node not found (ID: {parent_id})")
        print("-" * 50)

text_elements = [el for el in elements if el.category == "Text"]
table_elements = [el for el in elements if el.category == "Table"]
