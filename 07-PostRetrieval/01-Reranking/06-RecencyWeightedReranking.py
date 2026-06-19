from datetime import datetime, timedelta
import faiss
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

"""
Recency-weighted reranking algorithm implementation

TimeWeightedVectorStoreRetriever is a retriever that takes time into account, using document recency as an important ranking factor.

Core principles:
1. Time decay: a document's relevance score decays as time passes
2. Access updates: each time a document is accessed, its "last accessed time" is updated
3. Combined scoring: semantic similarity and time freshness are combined for overall ranking

Mathematical model:
- final_score = semantic_score * time_decay_factor
- time_decay_factor = exp(-decay_rate * time_since_last_access)

Technical features:
- Recency awareness: prioritizes documents that were most recently accessed or created
- Dynamic updates: a document's time weight is dynamically adjusted based on access patterns
- Decay control: an adjustable decay rate adapts to different application needs

Applicable scenarios:
- News retrieval: the latest news matters more
- Knowledge base maintenance: recently updated documents are more reliable
- Trend analysis: focuses on the latest data and information
- Real-time systems: applications that need to account for information recency

Parameter notes:
- decay_rate: the decay rate, controlling how strongly time affects relevance
- k: the number of documents returned
- last_accessed_at: the document's last accessed time
"""

print("🔄 Initializing recency-weighted reranking system...")

# 1. Configure the embedding model
print("📥 Configuring OpenAI embedding model...")
embeddings_model = OpenAIEmbeddings()
print("  Model: OpenAI Embeddings")
print("  Dimensions: 1536-dimensional vectors")
print("  Note: requires the OPENAI_API_KEY environment variable to be set")

# 2. Initialize the FAISS vector store
print(f"\n🏗️  Initializing vector store system...")
print("  📊 Creating FAISS index (L2 distance)...")
index = faiss.IndexFlatL2(1536)  # OpenAI embeddings have a dimension of 1536
print(f"    Index type: IndexFlatL2")
print(f"    Vector dimensions: 1536")

print("  🗄️  Configuring document store...")
vectorstore = FAISS(
    embeddings_model,           # Embedding model
    index,                      # FAISS index
    InMemoryDocstore({}),       # In-memory document store
    {}                          # Mapping from index to document ID
)
print("  ✅ Vector store system initialization complete")

# 3. Create the recency-weighted retriever
print(f"\n⏰ Creating recency-weighted retriever...")
print("  ⚙️  Retriever configuration parameters:")
decay_rate = 0.5
k_value = 1
print(f"    - decay_rate: {decay_rate} (decay rate; the larger the value, the stronger the time effect)")
print(f"    - k: {k_value} (number of documents returned)")
print("  📈 Decay mechanism explanation:")
print("    - Decay formula: score = semantic_score * exp(-decay_rate * hours_passed)")
print("    - A decay rate of 0.5 means document weight decays by about 39% per hour")

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=decay_rate,      # Decay rate: controls how strongly time affects relevance
    k=k_value                   # Number of documents returned
)
print("  ✅ Recency-weighted retriever created")

# 4. Prepare test documents
print(f"\n📋 Preparing test documents...")

# First document: set its access time to yesterday
yesterday = datetime.now() - timedelta(days=1)
print(f"  📅 Setting document 1's access time to yesterday: {yesterday.strftime('%Y-%m-%d %H:%M:%S')}")

print("  📄 Adding the first document (accessed yesterday)...")
doc1 = Document(
    page_content="hello world",
    metadata={"last_accessed_at": yesterday, "doc_id": "doc_1", "topic": "greeting"}
)
retriever.add_documents([doc1])
print(f"    Content: {doc1.page_content}")
print(f"    Access time: {yesterday.strftime('%Y-%m-%d %H:%M:%S')}")

# Second document: accessed at the current time (default)
current_time = datetime.now()
print(f"\n  📅 The second document will use the current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

print("  📄 Adding the second document (current time)...")
doc2 = Document(
    page_content="hello foo",
    metadata={"doc_id": "doc_2", "topic": "greeting"}
)
retriever.add_documents([doc2])
print(f"    Content: {doc2.page_content}")
print("    Access time: current time (default)")

# 5. Perform retrieval and recency analysis
print(f"\n🔍 Performing recency-weighted retrieval...")
query = "hello world"
print(f"Query: {query}")

print(f"\n  🧠 Recency-weighted retrieval process:")
print("    1. Compute semantic similarity between the query and each document")
print("    2. Compute the time decay factor based on each document's last accessed time")
print("    3. Combine the semantic score and time factor into a final score")
print("    4. Sort documents by final score in descending order")

print(f"\n  ⏳ Performing retrieval...")
results = retriever.get_relevant_documents(query)

# 6. Analyze and display the results
print(f"\n📊 Recency-weighted retrieval result analysis:")
print(f"{'='*60}")
print(f"🏆 Retrieval results")
print(f"{'='*60}")
print(f"Query: {query}")
print(f"Number of documents returned: {len(results)}")

for i, doc in enumerate(results, 1):
    print(f"\n📄 Rank {i}:")
    print(f"   Document content: {doc.page_content}")
    print(f"   Document ID: {doc.metadata.get('doc_id', 'unknown')}")
    print(f"   Topic: {doc.metadata.get('topic', 'unknown')}")

    # Analyze the impact of recency
    if 'last_accessed_at' in doc.metadata:
        access_time = doc.metadata['last_accessed_at']
        time_diff = datetime.now() - access_time
        hours_passed = time_diff.total_seconds() / 3600
        decay_factor = 1.0 / (1.0 + decay_rate * hours_passed)  # Simplified decay calculation

        print(f"   Last accessed: {access_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Time elapsed: {time_diff}")
        print(f"   Hours passed: {hours_passed:.2f}")
        print(f"   Time decay factor: {decay_factor:.4f}")
    else:
        print(f"   Last accessed: current time (just added)")
        print(f"   Time decay factor: 1.0000 (no decay)")

print(f"\n💡 Interpretation of results:")
if len(results) > 0:
    first_doc = results[0]
    if "foo" in first_doc.page_content:
        print("  ✅ 'hello foo' ranks first")
        print("  📈 Reason: although its semantic similarity to the query 'hello world' may be lower,")
        print("       its total score is higher because it is the most recently added document (high time weight)")
    else:
        print("  ✅ 'hello world' ranks first")
        print("  📈 Reason: semantic similarity is high enough to outweigh the effect of time decay")

# 7. Time simulation experiment
print(f"\n🔬 Time simulation experiment...")
print("  📅 Simulating retrieval results several hours later...")

# Use mock to simulate a future time
from langchain_core.utils import mock_now
import datetime as dt

# Simulate retrieval 8 hours later
future_time = dt.datetime(2028, 8, 8, 12, 0)  # Simulated future time
print(f"  ⏰ Simulated time: {future_time.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"  🔍 Performing retrieval at the simulated time...")
with mock_now(future_time):
    future_results = retriever.get_relevant_documents(query)

print(f"\n📊 Simulated-time retrieval results:")
print(f"Query: {query}")
print(f"Simulated time: {future_time.strftime('%Y-%m-%d %H:%M:%S')}")

for i, doc in enumerate(future_results, 1):
    print(f"\n📄 Rank {i} (simulated):")
    print(f"   Document content: {doc.page_content}")
    print(f"   Document ID: {doc.metadata.get('doc_id', 'unknown')}")

    # Compute decay under the simulated time
    if 'last_accessed_at' in doc.metadata:
        access_time = doc.metadata['last_accessed_at']
        time_diff = future_time - access_time
        hours_passed = time_diff.total_seconds() / 3600
        print(f"   Simulated time elapsed: {time_diff}")
        print(f"   Simulated hours passed: {hours_passed:.2f}")
    else:
        # For the document just added, compute the interval from addition to the simulated time
        time_diff = future_time - current_time
        hours_passed = time_diff.total_seconds() / 3600
        print(f"   Simulated time elapsed: {time_diff}")
        print(f"   Simulated hours passed: {hours_passed:.2f}")

print(f"\n📋 Recency-weighted reranking summary:")
print("- ✅ Recency awareness: prioritizes documents most recently accessed or created")
print("- ✅ Dynamic weighting: document importance is dynamically adjusted over time")
print("- ✅ Adjustable control: the decay_rate parameter controls how strongly time affects results")
print("- ✅ Combined scoring: balances semantic relevance and time freshness")
print("- 📈 Applicable scenarios: news retrieval, knowledge base maintenance, real-time systems")
print("- 🔧 Parameter tuning: adjust the decay rate and number of returned documents based on the application")
print("- ⚠️  Caveats: time metadata needs to be set sensibly")
print("- 💡 Best practice: combine with other retrieval methods to form a multi-stage retrieval pipeline")