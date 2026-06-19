"""
Note: before running this code, make sure the OpenAI API key is set as an environment variable.
On Linux/Mac, you can set it with:
export OPENAI_API_KEY='your-api-key'

On Windows, you can set it with:
set OPENAI_API_KEY=your-api-key

If you can't get an OpenAI API key, that's fine too - we have alternative options, please check the other programs.
"""

# Line 1: import the relevant libraries
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# Line 2: load the data
documents = SimpleDirectoryReader(input_files=["99-EN/black-myth-wukong/black_myth_wukong_setting.txt"]).load_data()
# Line 3: build the index
index = VectorStoreIndex.from_documents(documents)
# Line 4: create the query engine
query_engine = index.as_query_engine()
# Line 5: start asking questions
print(query_engine.query("What combat tools are there in Black Myth: Wukong?"))
