"""
Note: before running this code, make sure the OpenAI API key is set.
Add it to a .env file in the project root:
OPENAI_API_KEY=your-api-key

If you can't get an OpenAI API key, that's fine too - we have alternative options, please check the other programs.
"""

# Line 1: import the relevant libraries
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load OPENAI_API_KEY from the .env file into the environment
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Line 2: load the data
documents = SimpleDirectoryReader(input_files=["../99-EN/black-myth-wukong/black_myth_wukong_setting.txt"]).load_data()
# Line 3: build the index
index = VectorStoreIndex.from_documents(documents)
# Line 4: create the query engine
query_engine = index.as_query_engine()
# Line 5: start asking questions
print(query_engine.query("What combat tools are there in Black Myth: Wukong?"))
