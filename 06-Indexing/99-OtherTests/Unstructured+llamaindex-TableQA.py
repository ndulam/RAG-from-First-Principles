import os
from typing import List
from unstructured.partition.pdf import partition_pdf
import pandas as pd

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

# Global settings
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ---------------------------
# 1. Parse the PDF structure and extract text and tables
# ---------------------------
file_path = "90-Data/ComplexPDF/billionaires_page-1-5.pdf"  # Change this to your file path

elements = partition_pdf(
    file_path,
    strategy="hi_res",  # Use the high-resolution strategy
    extract_tables_in_paragraphs=True,  # Extract tables within paragraphs
    include_metadata=True  # Include metadata information
)  # Parse the PDF document

# Create a mapping from element ID to element
element_map = {element.id: element for element in elements if hasattr(element, 'id')}

for element in elements:
    if element.category == "Table":
        print("\nTable data:")
        print("Table metadata:", vars(element.metadata))  # Use vars() to show all metadata attributes
        print("Table content:")
        print(element.text)  # Print the table's text content

        # Get and print the parent node information
        parent_id = getattr(element.metadata, 'parent_id', None)
        if parent_id and parent_id in element_map:
            parent_element = element_map[parent_id]
            print("\nParent node information:")
            print(f"Type: {parent_element.category}")
            print(f"Content: {parent_element.text}")
            if hasattr(parent_element, 'metadata'):
                print(f"Parent node metadata: {vars(parent_element.metadata)}")  # Also use vars() to show all metadata
        else:
            print(f"Parent node not found (ID: {parent_id})")
        print("-" * 50)

text_elements = [el for el in elements if el.category == "Text"]
table_elements = [el for el in elements if el.category == "Table"]

# ---------------------------
# 2. Identify the year of each table
# ---------------------------
def extract_year_from_text(text):
    """Extract a year from text (simple match for the 1900-2099 pattern)"""
    import re
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return match.group(0) if match else None

table_data = []
last_seen_year = None

for element in elements:
    if element.category == "Text":
        extracted_year = extract_year_from_text(element.text)
        if extracted_year:
            last_seen_year = extracted_year  # Record the most recent year
    elif element.category == "Table":
        # Parse the table's text content
        rows = element.text.strip().split('\n')
        header = rows[0].split()  # Assume the first row is the header
        data = [row.split() for row in rows[1:]]
        df = pd.DataFrame(data, columns=header)
        table_data.append({"table": df, "year": last_seen_year})  # Associate the table with its year

# ---------------------------
# 3. Create the Pandas Query Engine and query
# ---------------------------
llm_for_table = OpenAI(model="gpt-4")

df_query_engines = [
    PandasQueryEngine(table_info["table"], llm=llm_for_table)
    for table_info in table_data
]

# Test query: find the net worth of the second-richest billionaire in 2023
for idx, engine in enumerate(df_query_engines):
    year = table_data[idx]["year"]
    if year == "2023":  # Only query the 2023 table
        response = engine.query("Who is the second richest billionaire?")
        print(f"Year: {year}, Response: {response}")

# ---------------------------
# 4. Build a vector index and perform retrieval
# ---------------------------
table_summaries = []
for idx, table_info in enumerate(table_data):
    table_text = table_info["table"].to_csv(index=False)
    year = table_info["year"]
    prompt = f"Summarize the main content of the {year} table in one sentence:\n\n{table_text}\n\nSummary:"
    summary = llm_for_table.complete(prompt).text.strip()
    table_summaries.append(summary)
    print(f"Auto-generated summary for the {year} table:", summary)

df_nodes = [
    IndexNode(text=table_summaries[idx], index_id=f"table_{idx}")
    for idx in range(len(table_summaries))
]

vector_index = VectorStoreIndex(df_nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

query_engine = RetrieverQueryEngine.from_args(
    vector_retriever, response_synthesizer=get_response_synthesizer(response_mode="compact")
)

# ---------------------------
# 5. Test a query that involves a year
# ---------------------------
query = "Who was the second richest billionaire in 2023?"
response = query_engine.query(query)
print("Query:", query)
print("Response:", response)
