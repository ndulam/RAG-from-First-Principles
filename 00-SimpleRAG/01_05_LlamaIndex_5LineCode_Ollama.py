"""
Use Ollama to run a large language model locally, no OpenAI API key needed.

1. Install the Ollama server:
   - Windows: visit https://ollama.com/download to download the installer
   - Linux/Mac: run curl -fsSL https://ollama.com/install.sh | sh

2. Download and run a model:
   - Open a terminal and run one of the following to download a model:
     ollama pull qwen:7b  # download the Qwen 7B model
     # or
     ollama pull llama2:7b  # download the Llama2 7B model
     # or
     ollama pull mistral:7b  # download the Mistral 7B model

3. Set an environment variable:
   - Add to your .env file:
     OLLAMA_MODEL=qwen:7b  # or another downloaded model name
"""

# Line 1: import the relevant libraries
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama # requires pip install llama-index-llms-ollama
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load a local embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

# Create the Ollama LLM, default URL: http://localhost:11434
llm = Ollama(
    model=os.getenv("OLLAMA_MODEL"),
    request_timeout=300.0
)

# Line 2: load the data
documents = SimpleDirectoryReader(input_files=["../99-EN/black-myth-wukong/black_myth_wukong_setting.txt"]).load_data()

# Line 3: build the index
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# Line 4: create the query engine
query_engine = index.as_query_engine(
    llm=llm
)

# Line 5: start asking questions
print(query_engine.query("What combat tools are there in Black Myth: Wukong?"))
