# Two-tier retrieval - Billionaires Ranking - requires pip install openpyxl
import os
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize the embedding model
embedding_function = SentenceTransformer(
    'BAAI/bge-m3',
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    trust_remote_code=True
)

# Connect to Milvus
client = MilvusClient("richman_bge_m3.db")

# 1. Create the summary vector collection
summary_collection_name = "billionaires_summary"
summary_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=100)
]

summary_schema = CollectionSchema(summary_fields, "Billionaires ranking yearly summary")
if not client.has_collection(summary_collection_name):
    client.create_collection(
        collection_name=summary_collection_name,
        schema=summary_schema
    )

# 2. Create the details vector collection
details_collection_name = "billionaires_details"
details_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="rank", dtype=DataType.INT64),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="wealth", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="company", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="industry", dtype=DataType.VARCHAR, max_length=100)
]

details_schema = CollectionSchema(details_fields, "Billionaires ranking detail information")
if not client.has_collection(details_collection_name):
    client.create_collection(
        collection_name=details_collection_name,
        schema=details_schema
    )

# 3. Load the Excel file and prepare the data
excel_file = "90-Data/ComplexPDF/TopTenBillionaires/WorldTopTenBillionaires.xlsx"

# Read every sheet in the Excel file and insert the data
with pd.ExcelFile(excel_file) as xls:
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            logging.info(f"Processing sheet: {sheet_name}")
            logging.info(f"Column names: {df.columns.tolist()}")

            # Normalize column names (remove line breaks and extra whitespace)
            df.columns = [col.replace('\n', ' ').strip() for col in df.columns]

            # Use the new column name format
            if 'Net_Worth' not in df.columns or 'Name' not in df.columns:
                raise ValueError(f"Required column not found: Net_Worth or Name")

            # Insert the summary data
            summary_embedding = embedding_function.encode([sheet_name])[0]
            
            client.insert(
                collection_name=summary_collection_name,
                data=[{
                    "vector": summary_embedding.tolist(),
                    "summary": sheet_name,
                    "table_name": sheet_name
                }]
            )
            
            # Insert the details data
            for _, row in df.iterrows():
                # Clean and format the data
                name = str(row['Name']).strip()
                wealth = str(row['Net_Worth']).strip()
                nationality = str(row['Nationality']).strip()
                source = str(row['Source']).strip()
                
                detail_text = f"{name} {wealth} {nationality} {source}"
                detail_embedding = embedding_function.encode([detail_text])[0]
                
                client.insert(
                    collection_name=details_collection_name,
                    data=[{
                        "vector": detail_embedding.tolist(),
                        "table_name": sheet_name,
                        "rank": int(row['No']),
                        "name": name,
                        "wealth": wealth,
                        "company": source,
                        "industry": nationality
                    }]
                )
            
            logging.info(f"Successfully processed sheet: {sheet_name}")

        except Exception as e:
            logging.error(f"Error processing sheet {sheet_name}: {str(e)}")
            logging.error(f"Error details: {e.__class__.__name__}")
            continue

# 4. Create the indexes
# Drop any existing index (if present)
try:
    client.drop_index(collection_name=summary_collection_name, index_name="vector")
except Exception as e:
    logging.warning(f"Error dropping summary index: {str(e)}")

try:
    client.drop_index(collection_name=details_collection_name, index_name="vector")
except Exception as e:
    logging.warning(f"Error dropping details index: {str(e)}")

# Create new indexes
try:
    # Use prepare_index_params to build the index parameters
    summary_index_params = client.prepare_index_params()
    summary_index_params.add_index(
        field_name="vector",  # The field to index, here the vector field
        index_type="IVF_FLAT",  # Index type
        metric_type="COSINE",  # Use cosine similarity as the vector similarity metric
        params={"nlist": 1024}  # Index parameters
    )

    client.create_index(
        collection_name=summary_collection_name,
        index_params=summary_index_params
    )
    logging.info("Successfully created the summary index")
except Exception as e:
    logging.error(f"Error creating the summary index: {str(e)}")

try:
    # Use prepare_index_params to build the index parameters
    details_index_params = client.prepare_index_params()
    details_index_params.add_index(
        field_name="vector",  # The field to index, here the vector field
        index_type="IVF_FLAT",  # Index type
        metric_type="COSINE",  # Use cosine similarity as the vector similarity metric
        params={"nlist": 1024}  # Index parameters
    )

    client.create_index(
        collection_name=details_collection_name,
        index_params=details_index_params
    )
    logging.info("Successfully created the details index")
except Exception as e:
    logging.error(f"Error creating the details index: {str(e)}")

# Load the collections so the indexes take effect
try:
    client.load_collection(summary_collection_name)
    client.load_collection(details_collection_name)
    logging.info("Successfully loaded the collections")
except Exception as e:
    logging.error(f"Error loading the collections: {str(e)}")

def search_relevant_table(question):
    # First-tier retrieval: search the summary collection for the most relevant sheet
    query_embedding = embedding_function.encode([question])[0]

    # Use the correct search parameter format
    summary_results = client.search(
        collection_name=summary_collection_name,
        data=[query_embedding.tolist()],
        limit=1,
        output_fields=["summary", "table_name"],
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
    )

    # Debug output
    logging.info(f"Summary search results: {summary_results}")

    if not summary_results or not summary_results[0]:
        return None, None

    # Check the result structure and extract the fields
    try:
        result_item = summary_results[0][0]
        logging.info(f"Result item keys: {result_item.keys()}")
        matched_summary = result_item.get('entity', {}).get('summary') or result_item.get('summary')
        matched_table = result_item.get('entity', {}).get('table_name') or result_item.get('table_name')

        if not matched_summary or not matched_table:
            logging.error(f"Unable to extract summary or table_name from the result: {result_item}")
            return None, None
    except Exception as e:
        logging.error(f"Error processing the summary result: {str(e)}")
        return None, None

    # Second-tier retrieval: search the details collection for specific information
    details_results = client.search(
        collection_name=details_collection_name,
        data=[query_embedding.tolist()],
        filter=f"table_name == '{matched_table}'",
        limit=5,
        output_fields=["rank", "name", "wealth", "company", "industry"],
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
    )

    # Debug output
    logging.info(f"Details search results structure: {details_results}")

    return matched_summary, details_results

def generate_answer(question):
    # Retrieve the relevant information
    summary, details = search_relevant_table(question)

    if not summary or not details:
        return "Sorry, no relevant information was found."

    # Build the prompt
    details_text = "\n".join([
        f"Rank: {result.get('entity', {}).get('rank')}, Name: {result.get('entity', {}).get('name')}, Wealth: {result.get('entity', {}).get('wealth')}, "
        f"Company: {result.get('entity', {}).get('company')}, Industry: {result.get('entity', {}).get('industry')}"
        for result in details[0]
    ])

    prompt = f"""Answer the question based on the following reference information:

Table description: {summary}

Relevant data:
{details_text}

Question: {question}

Please give a detailed answer based on the information above:"""

    # Use DeepSeek to generate the answer
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=1024
    )

    return response.choices[0].message.content

# Test example
if __name__ == "__main__":
    test_question = "Who was the world's richest person in 2023? How much was his wealth?"
    answer = generate_answer(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")