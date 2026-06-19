# Import required libraries
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
load_dotenv()

"""
Cohere Reranking Algorithm Implementation

Cohere Rerank is a commercial-grade reranking API service provided by Cohere, built on advanced language model technology.

Core features:
1. Enterprise-grade performance: based on large-scale pretrained models with strong semantic understanding
2. Multilingual support: handles reranking tasks across many languages, including Chinese
3. Ready to use: no need to deploy models locally, just call the API
4. Continuous optimization: models are updated continuously, with performance improving over time

Technical advantages:
- High precision: based on advanced Transformer architecture and large-scale training data
- Low latency: optimized inference engine supporting real-time reranking needs
- Easy integration: standard REST API interface, easy to integrate into existing systems
- Scalable: supports parallel reranking of large batches of documents

Suitable scenarios:
- Commercial-grade search systems
- Applications with high precision requirements
- Rapid prototyping and testing
- Multilingual retrieval systems

Cost considerations:
- Billed per API call
- Suitable for small-to-medium scale applications or cost-insensitive scenarios
- Recommended to evaluate cost during the development phase
"""

print("🔄 Initializing Cohere reranking service...")

# 1. API key configuration
print("🔐 Configuring Cohere API key...")
print("📝 Get your API key here: https://dashboard.cohere.com/api-keys")

# Get the Cohere API key - two configuration methods
import os

# Method 1: read from environment variable (recommended)
api_key_from_env = os.getenv('CO_API_KEY')

# Method 2: set directly (for testing only; use environment variables in production)
api_key = 'XXXX'  # Replace with your actual API key
os.environ['CO_API_KEY'] = api_key

if api_key_from_env:
    print("✅ Successfully read API key from environment variable")
else:
    print("⚠️  Using hardcoded API key (use an environment variable in production)")

print("🔒 Security reminder: keep your API key safe and never commit it to a code repository")

# 2. Prepare sample documents
print("\n📋 Preparing test documents...")
documents = [
    Document(
        page_content="Mount Wutai is one of China's four sacred Buddhist mountains, renowned as the dharma seat of Manjushri Bodhisattva.",
        metadata={"source": "Shanxi Travel Guide", "category": "Buddhist Culture", "location": "Xinzhou City"}
    ),
    Document(
        page_content="Yungang Grottoes is one of China's three major grotto complexes, famous for its exquisite Buddhist sculptures.",
        metadata={"source": "Shanxi Travel Guide", "category": "Grotto Art", "location": "Datong City"}
    ),
    Document(
        page_content="Pingyao Ancient City is one of China's best-preserved ancient county towns and is listed as a UNESCO World Heritage Site.",
        metadata={"source": "Shanxi Travel Guide", "category": "Ancient Architecture", "location": "Jinzhong City"}
    )
]

print(f"Number of documents: {len(documents)}")
for i, doc in enumerate(documents, 1):
    print(f"  Document {i}:")
    print(f"    Content: {doc.page_content}")
    print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"    Category: {doc.metadata.get('category', 'Unknown')}")
    print(f"    Location: {doc.metadata.get('location', 'Unknown')}")

# 3. Create a BM25 retriever (as initial retrieval)
print(f"\n🔍 Creating initial BM25 retriever...")
print("  BM25 is used for first-stage retrieval, providing a candidate document set")
retriever = BM25Retriever.from_documents(documents)
retriever.k = 3  # Set to return the top 3 results
print(f"✅ BM25 retriever configured, returning Top-{retriever.k} results")

# 4. Set up the Cohere reranker
print(f"\n🤖 Configuring Cohere reranker...")
reranker = CohereRerank(
    model="rerank-multilingual-v3.0"  # Multilingual reranking model, supports Chinese
)
print(f"Using model: rerank-multilingual-v3.0")
print("  Model features:")
print("  - ✅ Supports multiple languages (including Chinese)")
print("  - ✅ Based on advanced Transformer architecture")
print("  - ✅ Specifically optimized for reranking tasks")
print("  - ✅ Continuously updated and improved")

# 5. Execute query and reranking
print(f"\n🎯 Starting query and reranking...")
query = "What are the famous tourist attractions in Shanxi?"
print(f"Query: {query}")

print(f"\nStage 1 - Initial BM25 retrieval:")
print("  🔍 Performing initial retrieval using the BM25 algorithm...")
initial_docs = retriever.invoke(query)
print(f"  📊 BM25 retrieved {len(initial_docs)} candidate documents")

for i, doc in enumerate(initial_docs, 1):
    print(f"    {i}. {doc.page_content}")

print(f"\nStage 2 - Cohere reranking:")
print("  🤖 Calling the Cohere API for semantic reranking...")
print("  ⏳ Processing (this may take a few seconds)...")

try:
    # Use the Cohere reranker to rerank the BM25 results
    reranked_docs = reranker.compress_documents(
        documents=initial_docs,
        query=query
    )
    print("  ✅ Cohere reranking complete")

    # 6. Output reranking results
    print(f"\n{'='*60}")
    print(f"🏆 Final Cohere reranking results")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"\nReranked results (in descending order of relevance):")

    for i, doc in enumerate(reranked_docs, 1):
        print(f"\n📄 Rank {i}:")
        print(f"   Document content: {doc.page_content}")

        # Display document metadata
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"   Document source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Attraction category: {doc.metadata.get('category', 'Unknown')}")
            print(f"   Location: {doc.metadata.get('location', 'Unknown')}")

        # If a rerank score is available, display it
        if hasattr(doc, 'score'):
            print(f"   Rerank score: {doc.score:.4f}")

except Exception as e:
    print(f"  ❌ Cohere API call failed: {str(e)}")
    print("  💡 Possible reasons:")
    print("    - Invalid or expired API key")
    print("    - Network connection issue")
    print("    - API quota exhausted")
    print("    - Please check your API key configuration and network status")

print(f"\n📋 Cohere reranking summary:")
print("- ✅ Enterprise-grade performance: based on large-scale pretrained models")
print("- ✅ Multilingual support: natively supports Chinese and many other languages")
print("- ✅ Ready to use: no local deployment needed, just call the API")
print("- ✅ Continuous optimization: models are updated regularly, with performance steadily improving")
print("- 💰 Cost considerations: billed per API call")
print("- 🔐 Security reminder: keep your API key safe")
print("- 📈 Suitable scenarios: commercial-grade search, rapid prototyping, multilingual applications")