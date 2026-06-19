# Two-tier retrieval - Billionaires Ranking - requires pip install openpyxl
import os
from dotenv import load_dotenv
import pandas as pd
import logging
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Configure global settings
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 3. Load the Excel file and prepare the data
excel_file = "90-Data/ComplexPDF/TopTenBillionaires/WorldTopTenBillionaires.xlsx"

# Initialize the Node Parser
node_parser = SentenceSplitter(
    chunk_size=1024,  # Size of each chunk
    chunk_overlap=20,  # Overlap size between chunks
    include_metadata=True  # Include metadata
)

# Store the DataFrame and query engine for every table
table_dfs = []
df_query_engines = []
documents = []

# Read every sheet in the Excel file and insert the data
with pd.ExcelFile(excel_file) as xls:
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            logging.info(f"Processing sheet: {sheet_name}")

            # Convert the DataFrame to a string
            table_content = df.to_string(index=False)

            # Create a Document object
            doc = Document(
                text=table_content,
                metadata={"table_name": sheet_name}
            )
            documents.append(doc)

            # Store the DataFrame and create the query engine
            table_dfs.append(df)
            df_query_engine = PandasQueryEngine(df, llm=Settings.llm)
            df_query_engines.append(df_query_engine)

            logging.info(f"Successfully processed sheet: {sheet_name}")

        except Exception as e:
            logging.error(f"Error processing sheet {sheet_name}: {str(e)}")
            logging.error(f"Error details: {e.__class__.__name__}")
            continue

# Create the IndexNode objects
summaries = [
    f"This node provides information about the world's richest billionaires in {sheet_name}"
    for sheet_name in xls.sheet_names
]

df_nodes = [
    IndexNode(text=summary, index_id=f"pandas{idx}") # Details for each table
    for idx, summary in enumerate(summaries)
]

# Create the query engine mapping
df_id_query_engine_mapping = {
    f"pandas{idx}": df_query_engine
    for idx, df_query_engine in enumerate(df_query_engines)
}

# Create the vector index
vector_index = VectorStoreIndex(documents + df_nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# Create the recursive retriever
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_id_query_engine_mapping,
    verbose=True,
)

# Create the response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="compact")

# Create the query engine
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever, response_synthesizer=response_synthesizer
)

def generate_answer(question):
    # Use the query engine to generate the answer
    response = query_engine.query(question)
    return str(response)

# Test example
if __name__ == "__main__":
    test_question = "Who was the world's richest person in 2020? How much was his wealth?"
    answer = generate_answer(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")