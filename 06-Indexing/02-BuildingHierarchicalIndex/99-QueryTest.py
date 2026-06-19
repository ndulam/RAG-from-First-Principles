import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
from pymilvus import MilvusClient
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
client = MilvusClient("richman_bge_m3_v2.db")

def search_relevant_table(question):
    # First-layer retrieval: search for the most relevant sheet in the summary collection
    query_embedding = embedding_function.encode([question])[0]
    
    summary_results = client.search(
        collection_name="billionaires_summary",
        data=[query_embedding.tolist()],
        limit=1,
        output_fields=["table_name"],
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
    )
    
    if not summary_results or not summary_results[0]:
        return None, None
    
    matched_table = summary_results[0][0]['entity']['table_name']

    # Second-layer retrieval: search for specific information in the details collection
    details_results = client.search(
        collection_name="billionaires_details",
        data=[query_embedding.tolist()],
        filter=f"table_name == '{matched_table}'",
        limit=1,
        output_fields=["content"],
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
    )
    
    if not details_results or not details_results[0]:
        return None, None
    
    return matched_table, details_results[0][0]['entity']['content']

def generate_answer(question):
    # Retrieve relevant information
    table_name, content = search_relevant_table(question)

    if not table_name or not content:
        return "Sorry, no relevant information was found."

    # Build the prompt
    prompt = f"""Answer the question based on the following table information:

Table name: {table_name}

Table content:
{content}

Question: {question}

Please give a detailed answer based on the above information:"""

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

# List of test queries
test_queries = [
    # Basic queries
    "Who was the richest person in the world in 2023? What was their net worth?",
    "Who was the richest person in the world in 2020? What was their net worth?",
    "How many of the top ten on the 2022 world billionaires list were from the United States?",
    "How many of the top ten on the 2021 world billionaires list were from China?",

    # Comparative queries
    "What was the wealth gap between the richest and second-richest person in the world in 2020?",
    "Among the top ten on the 2019 world billionaires list, how does the number of billionaires from the tech industry compare to those from the luxury goods industry?",

    # Trend queries
    "Among the top ten on the 2019 world billionaires list, what proportion of the wealth belonged to billionaires from the tech industry?",
    "Among the top ten on the 2022 world billionaires list, who was the oldest billionaire?",

    # Complex queries
    "Among the top ten on the 2022 world billionaires list, what industries are the billionaires from Europe mainly engaged in?",
    "Among the top ten on the 2021 world billionaires list, what is the average age of billionaires whose wealth comes from the tech industry?"
]

# Run the tests
if __name__ == "__main__":
    print("Starting tests of the two-tier RAG system...\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}/{len(test_queries)}")
        print(f"Question: {query}")
        print("-" * 50)

        answer = generate_answer(query)
        print(f"Answer: {answer}")
        print("-" * 50) 