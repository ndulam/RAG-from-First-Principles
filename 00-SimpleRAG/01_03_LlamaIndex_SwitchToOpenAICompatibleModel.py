import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI  # import the OpenAI LLM class
from llama_index.embeddings.openai import OpenAIEmbedding # import the OpenAI Embedding class

# Load environment variables from the .env file
load_dotenv()

# --- Start configuring your custom API base URL and key ---
# Set CUSTOM_API_BASE_URL and CUSTOM_API_KEY in your .env file
custom_api_base_url = os.getenv("CUSTOM_API_BASE_URL")
custom_api_key = os.getenv("CUSTOM_API_KEY")

# (Optional) confirm the model names your third-party API supports/requires

# OpenAI default models:
llm_model_name = "gpt-4" # or another chat model your API supports
embedding_model_name = "text-embedding-ada-002" # or another embedding model your API supports

# Configure directly via code (recommended, clearer)
# Configure the global LLM (used for answer generation)
Settings.llm = OpenAI(
    model=llm_model_name,
    api_key=custom_api_key,
    api_base=custom_api_base_url,
    # Add any other parameters your API endpoint requires here
    # e.g.: temperature=0.7
)

# Configure the global Embedding Model (used for text vectorization)
Settings.embed_model = OpenAIEmbedding(
    model=embedding_model_name,
    api_key=custom_api_key,
    api_base=custom_api_base_url,
    # Some embedding endpoints may also accept extra parameters
)

# --- End of configuration ---

# Line 1: import the relevant libraries (some already imported above)
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader (already imported)

# Line 2: load the data
# Make sure the file path "../99-EN/black-myth-wukong/black_myth_wukong_setting.txt" is correct and readable
try:
    documents = SimpleDirectoryReader(input_files=["../99-EN/black-myth-wukong/black_myth_wukong_setting.txt"]).load_data()
except Exception as e:
    print(f"Error loading document: {e}")
    print("Please check the file path and permissions.")
    exit()

# Line 3: build the index
# Since we've already configured the global llm and embed_model via Settings,
# VectorStoreIndex.from_documents() will use them automatically.
# Note: building the index mainly uses the embedding_model.
print("Building index...")
index = VectorStoreIndex.from_documents(documents)
print("Index built.")

# Line 4: create the query engine
# as_query_engine() automatically uses the llm configured in Settings (and the embedding_model to encode the query).
query_engine = index.as_query_engine()
print("Query engine created.")

# Line 5: start asking questions
question = "What combat tools are there in Black Myth: Wukong?"
print(f"\nQuerying: {question}")
response = query_engine.query(question)
print("\nAnswer:")
print(response)
