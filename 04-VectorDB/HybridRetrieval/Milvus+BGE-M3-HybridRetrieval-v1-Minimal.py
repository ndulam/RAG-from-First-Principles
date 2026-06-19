import json
import time
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker
)
from pymilvus.exceptions import MilvusException
import scipy.sparse # make sure scipy is installed

# 0. Configuration (for easy modification)
DATA_PATH = "/root/AI-BOX/code/rag/rag-in-action/90-Data/Chronicles of Godslaying/BattleScenes.json"
COLLECTION_NAME = "wukong_hybrid_v4" # use a new collection name to avoid conflicts with old data
MILVUS_URI = "./wukong_v4.db" # use a new database file
BATCH_SIZE = 50 # try reducing the batch size, e.g. 10 or 20, for testing
DEVICE = "cpu" # or "cuda" if you have a properly configured GPU

print("Script started...")

# 1. Load data
print(f"1. Loading data from {DATA_PATH}...")
try:
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
except FileNotFoundError:
    print(f"Error: data file {DATA_PATH} not found. Please check the path.")
    exit()
except json.JSONDecodeError:
    print(f"Error: data file {DATA_PATH} has invalid JSON format.")
    exit()

docs = []
metadata = []
for item in dataset.get('data', []): # use .get to avoid errors if the 'data' key is missing
    text_parts = [item.get('title', ''), item.get('description', '')]
    if 'combat_details' in item and isinstance(item['combat_details'], dict):
        text_parts.extend(item['combat_details'].get('combat_style', []))
        text_parts.extend(item['combat_details'].get('abilities_used', []))
    if 'scene_info' in item and isinstance(item['scene_info'], dict):
        text_parts.extend([
            item['scene_info'].get('location', ''),
            item['scene_info'].get('environment', ''),
            item['scene_info'].get('time_of_day', '')
        ])
    # filter out None and empty strings, then join
    docs.append(' '.join(filter(None, [str(part).strip() for part in text_parts if part])))
    metadata.append(item)

if not docs:
    print("Error: failed to load any documents from the data file. Please check the file content and structure.")
    exit()
print(f"Data loading complete, {len(docs)} documents total.")

# 2. Generate vectors
print("2. Generating vectors...")
try:
    ef = BGEM3EmbeddingFunction(use_fp16=False, device=DEVICE)
    docs_to_embed = docs
    print(f"Generating vectors for {len(docs_to_embed)} documents...")
    docs_embeddings = ef(docs_to_embed)
    print("Vector generation complete.")
    print(f"  Dense vector dimension: {ef.dim['dense']}")
    if "sparse" in docs_embeddings and docs_embeddings["sparse"].shape[0] > 0:
        print(f"  Sparse vector type (overall): {type(docs_embeddings['sparse'])}")
        #  print the shape and partial content of the first sparse vector for inspection
        first_sparse_vector_row_obj = docs_embeddings['sparse'][0] # this gives a sparse array object representing a single row
        print(f"  First sparse vector (row object type): {type(first_sparse_vector_row_obj)}")
        print(f"  First sparse vector (row object shape): {first_sparse_vector_row_obj.shape}")
        if hasattr(first_sparse_vector_row_obj, 'col') and hasattr(first_sparse_vector_row_obj, 'data'):
            print(f"  First sparse vector (partial column indices/col): {first_sparse_vector_row_obj.col[:5]}")
            print(f"  First sparse vector (partial data): {first_sparse_vector_row_obj.data[:5]}")
        elif hasattr(first_sparse_vector_row_obj, 'indices') and hasattr(first_sparse_vector_row_obj, 'data'): # Fallback for other types
            print(f"  First sparse vector (partial indices): {first_sparse_vector_row_obj.indices[:5]}")
            print(f"  First sparse vector (partial data): {first_sparse_vector_row_obj.data[:5]}")
        else:
            print("  Could not directly retrieve column indices and data attributes of the first sparse vector.")
    else:
        print("Warning: no sparse vectors were generated, or the sparse vector is empty.")

except Exception as e:
    print(f"Error occurred while generating vectors: {e}")
    exit()

# 3. Connect to Milvus
print(f"3. Connecting to Milvus (URI: {MILVUS_URI})...")
try:
    connections.connect(uri=MILVUS_URI)
    print("Successfully connected to Milvus.")
except MilvusException as e:
    print(f"Failed to connect to Milvus: {e}")
    exit()

# 4. Create collection
print(f"4. Preparing collection '{COLLECTION_NAME}'...")
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])
]
schema = CollectionSchema(fields, description="Wukong Hybrid Search Collection v4")

try:
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists, dropping it...")
        utility.drop_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' dropped successfully.")
        time.sleep(1)

    print(f"Creating collection '{COLLECTION_NAME}'...")
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    print(f"Collection '{COLLECTION_NAME}' created successfully.")

    print("Creating index for sparse_vector (SPARSE_INVERTED_INDEX, IP)...")
    collection.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    print("sparse_vector index created successfully.")
    time.sleep(0.5)

    print("Creating index for dense_vector (AUTOINDEX, IP)...")
    collection.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
    print("dense_vector index created successfully.")
    time.sleep(0.5)

    print(f"Loading collection '{COLLECTION_NAME}'...")
    collection.load()
    print(f"Collection '{COLLECTION_NAME}' loaded successfully.")

except MilvusException as e:
    print(f"Milvus error occurred while creating or loading the collection/index: {e}")
    exit()
except Exception as e:
    print(f"Unknown error occurred while creating or loading the collection/index: {e}")
    exit()

# 5. Insert data
print("5. Preparing to insert data...")
num_docs_to_insert = len(docs_to_embed)
try:
    for i in range(0, num_docs_to_insert, BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, num_docs_to_insert)
        batch_data = []
        print(f"  Preparing batch {i // BATCH_SIZE + 1} (index {i} to {end_idx - 1})...")

        for j in range(i, end_idx):
            item_metadata = metadata[j]

            # Key: convert the sparse vector format
            # when indexing a row from a csr_array, you may get a coo_array or other sparse format
            sparse_row_obj = docs_embeddings["sparse"][j]
            # coo_array uses .col and .data
            if hasattr(sparse_row_obj, 'col') and hasattr(sparse_row_obj, 'data'):
                milvus_sparse_vector = {int(idx_col): float(val) for idx_col, val in zip(sparse_row_obj.col, sparse_row_obj.data)}
            # csr_array (if it is directly a row csr_array) uses .indices and .data
            elif hasattr(sparse_row_obj, 'indices') and hasattr(sparse_row_obj, 'data'):
                 milvus_sparse_vector = {int(idx_col): float(val) for idx_col, val in zip(sparse_row_obj.indices, sparse_row_obj.data)}
            else:
                print(f"Warning: unrecognized sparse row object type {type(sparse_row_obj)} at index {j}. Skipping this entry.")
                continue # or raise an error

            doc_text = docs_to_embed[j]
            if len(doc_text) > 65530 : 
                doc_text = doc_text[:65530]

            title_text = item_metadata.get("title", "N/A")
            if len(title_text) > 500:
                title_text = title_text[:500]

            batch_data.append({
                "text": doc_text,
                "id": str(item_metadata.get("id", f"unknown_id_{j}")),
                "title": title_text,
                "category": item_metadata.get("category", "N/A"),
                "location": item_metadata.get("scene_info", {}).get("location", "N/A"),
                "environment": item_metadata.get("scene_info", {}).get("environment", "N/A"),
                "sparse_vector": milvus_sparse_vector,
                "dense_vector": docs_embeddings["dense"][j].tolist() 
            })
        
        if not batch_data: # if all sparse vectors in the batch could not be processed
            print(f"  Batch {i // BATCH_SIZE + 1} is empty, skipping insertion.")
            continue

        print(f"  Inserting batch {i // BATCH_SIZE + 1} ({len(batch_data)} records)...")
        insert_result = collection.insert(batch_data)
        print(f"  Batch {i // BATCH_SIZE + 1} inserted successfully, primary keys: {insert_result.primary_keys[:5]}...")
        collection.flush()
        print(f"  Batch {i // BATCH_SIZE + 1} flush complete.")
        time.sleep(0.5)

    print(f"All data inserted. Total of {collection.num_entities} entities.")

except MilvusException as e:
    print(f"Milvus error occurred while inserting data: {e}")
    if 'batch_data' in locals() and batch_data:
        print("First record (partial) of the problematic batch:")
        print(f"  Text: {batch_data[0]['text'][:100]}...")
        print(f"  ID: {batch_data[0]['id']}")
        print(f"  Title: {batch_data[0]['title']}")
    exit()
except Exception as e:
    print(f"Unknown error occurred while inserting data: {e}")
    if 'batch_data' in locals() and batch_data:
        print("First record (partial) of the problematic batch:")
        print(f"  Text: {batch_data[0]['text'][:100]}...")
    exit()


# 6. Hybrid search (example)
def hybrid_search(query, category=None, environment=None, limit=5, weights=None):
    if weights is None:
        weights = {"sparse": 0.5, "dense": 0.5}

    print(f"\n6. Performing hybrid search: '{query}'")
    print(f"   Category: {category}, Environment: {environment}, Limit: {limit}, Weights: {weights}")

    try:
        query_embeddings = ef([query])

        conditions = []
        if category:
            conditions.append(f'category == "{category}"')
        if environment:
            conditions.append(f'environment == "{environment}"')
        expr = " && ".join(conditions) if conditions else None
        print(f"   Filter expression: {expr}")
        
        search_params_dense = {"metric_type": "IP", "params": {}} 
        search_params_sparse = {"metric_type": "IP", "params": {}}

        if expr:
            search_params_dense["expr"] = expr
            search_params_sparse["expr"] = expr
        
        dense_req = AnnSearchRequest(
            data=[query_embeddings["dense"][0].tolist()], 
            anns_field="dense_vector",
            param=search_params_dense,
            limit=limit
        )
        
        # convert the query sparse vector format
        query_sparse_row_obj = query_embeddings["sparse"][0] # indexing returns a single-row sparse object
        if hasattr(query_sparse_row_obj, 'col') and hasattr(query_sparse_row_obj, 'data'):
            query_milvus_sparse_vector = {int(idx): float(val) for idx, val in zip(query_sparse_row_obj.col, query_sparse_row_obj.data)}
        elif hasattr(query_sparse_row_obj, 'indices') and hasattr(query_sparse_row_obj, 'data'):
            query_milvus_sparse_vector = {int(idx): float(val) for idx, val in zip(query_sparse_row_obj.indices, query_sparse_row_obj.data)}
        else:
            print(f"Error: unrecognized query sparse vector type {type(query_sparse_row_obj)}.")
            return []


        sparse_req = AnnSearchRequest(
            data=[query_milvus_sparse_vector], 
            anns_field="sparse_vector",
            param=search_params_sparse,
            limit=limit
        )
        
        rerank = WeightedRanker(weights["sparse"], weights["dense"])

        print("   Sending hybrid search request to Milvus...")
        results = collection.hybrid_search(
            reqs=[sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            output_fields=["text", "id", "title", "category", "location", "environment", "pk"]
        )

        print("   Search complete. Results:")
        if not results or not results[0]:
            print("   No results found.")
            return []

        processed_results = []
        for hit in results[0]:
            processed_results.append({
                "id": hit.entity.get("id"),
                "pk": hit.id, 
                "title": hit.entity.get("title"),
                "text_preview": hit.entity.get("text", "")[:200] + "...", 
                "category": hit.entity.get("category"),
                "location": hit.entity.get("location"),
                "environment": hit.entity.get("environment"),
                "distance": hit.distance 
            })
        return processed_results

    except MilvusException as e:
        print(f"Milvus error occurred during hybrid search: {e}")
        return []
    except Exception as e:
        print(f"Unknown error occurred during hybrid search: {e}")
        return []

# Example search calls
if collection.num_entities > 0:
    print("\nStarting example search...")
    search_results = hybrid_search("Wukong's combat techniques", category="War of Gods and Demons", limit=3)
    if search_results:
        for res in search_results:
            print(f"  - PK: {res['pk']}, Title: {res['title']}, Distance: {res['distance']:.4f}")
            print(f"    Category: {res['category']}, Location: {res['location']}")
            print(f"    Preview: {res['text_preview']}\n")

    search_results_filtered = hybrid_search("Battle at the Flaming Mountain", environment="Volcano", limit=2)
    if search_results_filtered:
        for res in search_results_filtered:
            print(f"  - PK: {res['pk']}, Title: {res['title']}, Distance: {res['distance']:.4f}")
            print(f"    Category: {res['category']}, Location: {res['location']}, Environment: {res['environment']}")
            print(f"    Preview: {res['text_preview']}\n")
else:
    print("\nNo entities in the collection, skipping example search.")

print("\nScript execution complete.")

