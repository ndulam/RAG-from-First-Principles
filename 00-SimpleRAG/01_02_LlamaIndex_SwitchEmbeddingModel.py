# Import the relevant libraries
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # requires pip install llama-index-embeddings-huggingface

# Load a local embedding model
# import os
# os.environ['HF_ENDPOINT']= 'https://hf-mirror.com' # set a mirror if HuggingFace happens to be blocked
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh" # model path/name (downloaded from HuggingFace on first run)
    )

# Load the data
documents = SimpleDirectoryReader(input_files=["../99-EN/black-myth-wukong/black_myth_wukong_setting.txt"]).load_data()

# Build the index
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# Create the query engine
query_engine = index.as_query_engine()

# Start asking questions
print(query_engine.query("What combat tools are there in Black Myth: Wukong?"))
