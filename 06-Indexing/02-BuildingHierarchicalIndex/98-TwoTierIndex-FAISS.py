import os
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

# Load environment variables
load_dotenv()

# 1. Prepare the table description data
table_descriptions = [
    "The 2023 WorldTopTenBillionaires list, showing the ten wealthiest people in the world that year and their net worth.",
    "The 2022 WorldTopTenBillionaires list, recording the ten wealthiest people in the world that year and their net worth.",
    "The 2021 WorldTopTenBillionaires list, showing the ten wealthiest people in the world that year and their net worth.",
    "The 2020 WorldTopTenBillionaires list, recording the ten wealthiest people in the world that year and their net worth.",
    "The 2019 WorldTopTenBillionaires list, showing the ten wealthiest people in the world that year and their net worth."
]

# 2. Set up the first-layer embedding model (used to match the year)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
desc_embeddings = model.encode(table_descriptions)

# 3. Create the first-layer vector store
dimension = desc_embeddings.shape[1]
desc_index = faiss.IndexFlatL2(dimension)
desc_index.add(desc_embeddings.astype('float32'))

# 4. Load the Excel file and prepare the second-layer data
excel_file = "90-Data/ComplexPDF/TopTenBillionaires/WorldTopTenBillionaires.xlsx"
all_tables_data = {}

# Read all sheets from the Excel file
with pd.ExcelFile(excel_file) as xls:
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # Convert the DataFrame into text format
        table_text = df.to_string(index=False)
        all_tables_data[sheet_name] = table_text

# 5. Create the second-layer vector store
table_embeddings = model.encode(list(all_tables_data.values()))
table_index = faiss.IndexFlatL2(dimension)
table_index.add(table_embeddings.astype('float32'))

def search_relevant_table(question):
    # First-layer retrieval: match the year
    query_embedding = model.encode([question])[0]
    distances, indices = desc_index.search(
        np.array([query_embedding]).astype('float32'),
        k=1
    )
    matched_year = indices[0][0]

    # Second-layer retrieval: search for specific information within the matched year's table
    table_embedding = model.encode([all_tables_data[f"billionaires_table_{matched_year+2}"]])[0]
    distances, indices = table_index.search(
        np.array([table_embedding]).astype('float32'),
        k=1
    )

    return table_descriptions[matched_year], all_tables_data[f"billionaires_table_{matched_year+2}"]

def generate_answer(question):
    # Retrieve relevant information
    year_context, table_context = search_relevant_table(question)

    # Build the prompt
    prompt = f"""Answer the question based on the following reference information:

Year information: {year_context}

Relevant data:
{table_context}

Question: {question}

Please give a detailed answer based on the above information:"""

    # Use DeepSeek to generate the answer
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
    test_question = "Who was the richest person in the world in 2023? What was their net worth?"
    answer = generate_answer(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")