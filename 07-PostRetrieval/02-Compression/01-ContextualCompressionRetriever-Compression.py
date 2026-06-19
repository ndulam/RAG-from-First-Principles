# Import required libraries
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
load_dotenv()

# Get the Cohere API key
# URL: https://dashboard.cohere.com/api-keys
# If CO_API_KEY isn't set in the env file, you can also set it this way
# import os
# api_key = 'XXXX'
# os.environ['CO_API_KEY'] = api_key

documents = [
    Document(
        page_content="Mount Wutai is one of China's four sacred Buddhist mountains, famous as the bodhimanda of Manjushri Bodhisattva.",
        metadata={"source": "Shanxi Travel Guide"}
    ),
    Document(
        page_content="Yungang Grottoes is one of China's three great grotto complexes, renowned for its exquisite Buddhist sculptures.",
        metadata={"source": "Shanxi Travel Guide"}
    ),
    Document(
        page_content="Pingyao Ancient City is one of China's best-preserved ancient county towns, listed as a UNESCO World Heritage Site.",
        metadata={"source": "Shanxi Travel Guide"}
    )
]
# Create a BM25 retriever
retriever = BM25Retriever.from_documents(documents)
retriever.k = 3  # Return the top 3 results
# Set up the Cohere reranker
# Model URL: https://huggingface.co/Cohere/rerank-multilingual-v3.0
compressor = CohereRerank(model="rerank-multilingual-v3.0")
# Create the ContextualCompressionRetriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
# Run the query, rerank, and compress
query = "What are some famous tourist attractions in Shanxi?"
compressed_docs = compression_retriever.invoke(query)
# Print the compressed results
print(f"Query: {query}\n")
print("Reranked and compressed results:")
for i, doc in enumerate(compressed_docs, 1):
    print(f"{i}. {doc.page_content}")

