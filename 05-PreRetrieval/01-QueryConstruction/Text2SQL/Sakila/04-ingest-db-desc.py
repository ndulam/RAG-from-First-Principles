# ingest_dbdesc.py
import logging
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
from pymilvus import model
import torch
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Initialize the embedding function
embedding_function = model.dense.OpenAIEmbeddingFunction(model_name='text-embedding-3-large')

# 2. Load the DB description (db_description.yaml is assumed to have the format
#    table_name:
#       column_name: "business meaning"
#       ...
# )
with open("90-Data/sakila/db_description.yaml", "r") as f:
    desc_map = yaml.safe_load(f)
    logging.info(f"[DBDESC] Loaded descriptions for {len(desc_map)} tables from the YAML file")

# 3. Connect to Milvus
client = MilvusClient("text2sql_milvus_sakila.db")

# 4. Define the collection schema
vector_dim = len(embedding_function(["dummy"])[0])
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="column_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),
]
schema = CollectionSchema(fields, description="DB Description Knowledge Base", enable_dynamic_field=False)

# 5. Create the collection (if it doesn't already exist)
collection_name = "dbdesc_knowledge"
if not client.has_collection(collection_name):
    client.create_collection(collection_name=collection_name, schema=schema)
    logging.info(f"[DBDESC] Created new collection {collection_name}")
else:
    logging.info(f"[DBDESC] Collection {collection_name} already exists")

# 6. Add an index to the vector field
index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE", params={"nlist": 1024})
client.create_index(collection_name=collection_name, index_params=index_params)

# 7. Bulk insert the descriptions
data = []
texts = []
for tbl, cols in desc_map.items():
    for col, desc in cols.items():
        texts.append(desc)
        data.append({"table_name": tbl, "column_name": col, "description": desc})

logging.info(f"[DBDESC] Preparing to process {len(data)} column descriptions")

# Generate all embeddings
embeddings = embedding_function(texts)
logging.info(f"[DBDESC] Successfully generated {len(embeddings)} vector embeddings")

# Organize into Milvus insert format
records = []
for emb, rec in zip(embeddings, data):
    rec["vector"] = emb
    records.append(rec)

res = client.insert(collection_name=collection_name, data=records)
logging.info(f"[DBDESC] Successfully inserted {len(records)} records into Milvus")
logging.info(f"[DBDESC] Insert result: {res}")

logging.info("[DBDESC] Knowledge base construction complete")
