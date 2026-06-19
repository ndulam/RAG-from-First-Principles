# Line 1: import the relevant libraries
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load a local embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

# Create the DeepSeek LLM
llm = DeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# Line 2: load the data
documents = SimpleDirectoryReader(input_files=["99-EN/black-myth-wukong/black_myth_wukong_setting.txt"]).load_data()

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
