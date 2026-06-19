# Import the relevant libraries
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # requires pip install llama-index-embeddings-huggingface
from llama_index.llms.deepseek import DeepSeek  # requires pip install llama-index-llms-deepseek

from llama_index.core import Settings # check what settings are available
# https://docs.llamaindex.ai/en/stable/examples/llm/deepseek/
# Settings.llm = DeepSeek(model="deepseek-chat")
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh")
# Settings.llm = OpenAI(model="gpt-3.5-turbo")
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Load environment variables
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Create the DeepSeek LLM (calls the latest DeepSeek large model via API)
llm = DeepSeek(
    model="deepseek-reasoner", # use the latest reasoning model, R1
    api_key=os.getenv("DEEPSEEK_API_KEY")  # get the API key from an environment variable
)

# Load the data
documents = SimpleDirectoryReader(input_files=["99-EN/black-myth-wukong/black_myth_wukong_setting.txt"]).load_data()

# Build the index
index = VectorStoreIndex.from_documents(
    documents,
    # llm=llm  # set the LLM used when building the index (usually not needed)
)

# Create the query engine
query_engine = index.as_query_engine(
    llm=llm  # set the generation model
    )

# Start asking questions
print(query_engine.query("What combat tools are there in Black Myth: Wukong?"))
