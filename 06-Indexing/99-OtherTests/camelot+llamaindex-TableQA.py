import os
from typing import List
import camelot

# Import relevant LlamaIndex modules
from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.file import PyMuPDFReader
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.schema import IndexNode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer

# Global settings: use GPT-3.5-turbo as the LLM, and a smaller OpenAIEmbedding model
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ---------------------------
# 1. Load the PDF body text (narrative text)
# ---------------------------
file_path = "90-Data/ComplexPDF/billionaires_page.pdf"
reader = PyMuPDFReader()
docs = reader.load(file_path)

# ---------------------------
# 2. Use camelot to extract tables from the PDF
# ---------------------------
def get_tables(path: str, pages: List[int]):
    table_dfs = []
    for page in pages:
        table_list = camelot.read_pdf(path, pages=str(page))
        if len(table_list) > 0:
            table_df = table_list[0].df
            # Use the first row as the header and reset the index
            table_df = (
                table_df.rename(columns=table_df.iloc[0])
                .drop(table_df.index[0])
                .reset_index(drop=True)
            )
            table_dfs.append(table_df)
    return table_dfs

# Assume page 3 is the 2023 billionaires ranking table and page 25 is the annual statistics table
table_dfs = get_tables(file_path, pages=[3, 25])

# ---------------------------
# 3. Create a Pandas Query Engine for each table
# ---------------------------
# It's recommended to use a stronger LLM for table queries; GPT-4 is used here (other models also work)
llm_for_table = OpenAI(model="gpt-4")
df_query_engines = [
    PandasQueryEngine(table_df, llm=llm_for_table) for table_df in table_dfs
]

# Test table queries
response_table0 = df_query_engines[0].query(
    "What's the net worth of the second richest billionaire in 2023?"
)
print("Table0 response:", str(response_table0))
response_table1 = df_query_engines[1].query(
    "How many billionaires were there in 2009?"
)
print("Table1 response:", str(response_table1))

# ---------------------------
# 4. Automatically generate a summary for each table
# ---------------------------
# For each table, convert it to CSV text, then call the LLM to generate a summary
table_summaries = []
for idx, table_df in enumerate(table_dfs):
    # Convert the table to text format (the format can be adjusted as needed)
    table_text = table_df.to_csv(index=False)
    prompt = (
        "Summarize the main content of the table below in one sentence, describing the information it presents:\n\n"
        f"{table_text}\n\nSummary:"
    )
    summary = llm_for_table.complete(prompt).text.strip()
    table_summaries.append(summary)
    print(f"Auto-generated summary for table {idx}:", summary)

# ---------------------------
# 5. Build a vector index (combining body text nodes and table summary nodes)
# ---------------------------
# 5.1 Build nodes from the PDF body text (using the default text splitter)
doc_nodes = Settings.node_parser.get_nodes_from_documents(docs)

# 5.2 Use the auto-generated summaries to create an IndexNode for each table, with a unique index_id
df_nodes = [
    IndexNode(text=table_summaries[idx], index_id=f"pandas{idx}")
    for idx in range(len(table_summaries))
]

# Build a mapping between IndexNode id and PandasQueryEngine
df_id_query_engine_mapping = {
    f"pandas{idx}": df_query_engine
    for idx, df_query_engine in enumerate(df_query_engines)
}

# Combine the body text nodes and table nodes to form the final index
all_nodes = doc_nodes + df_nodes
vector_index = VectorStoreIndex(all_nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# ---------------------------
# 6. Use RecursiveRetriever to build a hierarchical query engine
# ---------------------------
# When a table IndexNode is retrieved, further invoke the corresponding PandasQueryEngine
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_id_query_engine_mapping,
    verbose=True,
)

# Build the Response Synthesizer using compact mode
response_synthesizer = get_response_synthesizer(response_mode="compact")

# Construct the final RetrieverQueryEngine
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever, response_synthesizer=response_synthesizer
)

# ---------------------------
# 7. Execute queries to test the recursive retrieval results
# ---------------------------
# Example 1: query the net worth of the second-richest billionaire in 2023
query_1 = "What's the net worth of the second richest billionaire in 2023?"
response = query_engine.query(query_1)
print("Query:", query_1)
print("Response:", str(response))

# Example 2: query how many billionaires there were in 2009
query_2 = "How many billionaires were there in 2009?"
response = query_engine.query(query_2)
print("Query:", query_2)
print("Response:", str(response))

# Example 3: query exclusion rules (e.g. which billionaires are not in the list)
query_3 = "Which billionaires are excluded from this list?"
response = query_engine.query(query_3)
print("Query:", query_3)
print("Response:", str(response))
