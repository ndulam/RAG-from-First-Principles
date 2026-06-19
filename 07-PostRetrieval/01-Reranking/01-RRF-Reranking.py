# Import required libraries
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain.load import dumps, loads

"""
RRF (Reciprocal Rank Fusion) reranking algorithm implementation

RRF is a simple yet effective algorithm for fusing multiple retrieval result lists.
It fuses the rankings of results from multiple retrieval queries to improve
retrieval accuracy and coverage.

Core idea:
1. For a single user question, generate multiple queries from different angles
2. Retrieve results for each query separately
3. Use the RRF algorithm to fuse multiple retrieval result lists into one unified ranked list
4. The RRF algorithm assigns a score to each document: score = 1/(rank + k), where rank is the
   document's rank within a given result list

Advantages:
- Improves retrieval coverage: multiple queries can retrieve relevant documents from different angles
- Reduces the bias of a single query: fusing multiple queries reduces the limitations of any one query
- Simple and efficient: the algorithm has low complexity and is easy to implement and understand
"""

# Document directory configuration
doc_dir = "90-Data/Shanxi Cultural Tourism"

def load_documents(directory):
    """
    Document loading function

    Purpose: Read all documents (PDF and TXT formats supported) in the specified directory

    Args:
        directory (str): Path to the directory containing the documents

    Returns:
        list: List of loaded documents, each containing content and metadata

    Notes:
        - Iterates over all files in the directory
        - Picks the appropriate loader based on the file extension
        - Supports PDF and TXT format files
        - Skips unsupported file formats
    """
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        if filename.endswith(".pdf"):
            # Use PyPDFLoader to load PDF files
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            # Use TextLoader to load TXT files
            loader = TextLoader(filepath)
        else:
            continue  # Skip unsupported file types

        # Load the document and add it to the list
        documents.extend(loader.load())
    return documents

# Step 1: Load documents
print("📖 Loading documents...")
docs = load_documents(doc_dir)
print(f"✅ Successfully loaded {len(docs)} documents")

# Step 2: Text chunking (splitting)
print("\n🔪 Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # Maximum number of characters per text chunk
    chunk_overlap=50     # Overlapping characters between adjacent chunks, to keep context continuous
)
splits = text_splitter.split_documents(docs)
print(f"✅ Documents split into {len(splits)} text chunks")

# Step 3: Create the vector index
print("\n🔍 Creating vector index...")
# Use HuggingFace's lightweight embedding model
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Use the Chroma vector database to store document vectors
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
# Create the retriever
retriever = vectorstore.as_retriever()
print("✅ Vector index created")

def reciprocal_rank_fusion(results: list[list], k=60):
    """
    RRF (Reciprocal Rank Fusion) algorithm implementation

    Purpose: Fuse multiple retrieval result lists into one unified ranked list

    Args:
        results (list[list]): Multiple retrieval result lists, each containing documents sorted by relevance
        k (int): The RRF algorithm's tuning parameter, default value 60 (an empirical value)

    Returns:
        list: A list of (document, score) tuples after fusion, sorted in descending order of score

    Algorithm details:
        1. For each document in each retrieval result list
        2. Compute the document's RRF score: score = 1 / (rank + k)
        3. If the same document appears in multiple lists, accumulate its score
        4. Sort all documents by their final score

    Advantages:
        - The smaller the rank (i.e., the higher the document is ranked), the higher the score
        - The k parameter prevents the denominator from being 0 and tunes the gap between different ranks
        - Documents that appear multiple times accumulate a higher score
    """
    print(f"🔄 RRF algorithm processing {len(results)} retrieval result lists...")

    fused_scores = {}  # Stores the cumulative score for each document

    # Iterate over each retrieval result list
    for list_idx, docs in enumerate(results):
        print(f"  Processing result list {list_idx + 1}, containing {len(docs)} documents")

        # Iterate over each document in this list
        for rank, doc in enumerate(docs):
            # Serialize the document to a string to use as a unique identifier
            doc_str = dumps(doc)

            # If this document appears for the first time, initialize its score
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0

            # Compute the RRF score and accumulate it
            rrf_score = 1 / (rank + k)
            fused_scores[doc_str] += rrf_score

            # Debug info: show the document's rank and score within the current list
            if rank < 3:  # Only show details for the top 3 documents
                print(f"    Document {rank+1}: RRF score = 1/({rank}+{k}) = {rrf_score:.4f}")

    # Sort by score in descending order, returning a list of (document, score) tuples
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    print(f"✅ RRF fusion complete, {len(reranked_results)} unique documents in total")
    return reranked_results

# Step 4: Multi-query generation
print("\n💭 Configuring the multi-query generator...")
template = """You are an assistant that helps users generate multiple search queries.

Based on the following question, generate 4 related search queries from different angles. These queries should:
1. Understand the original question from different perspectives
2. Use different keywords and phrasing
3. Cover different aspects of the question

Original question: {question}

Please generate 4 related search queries:"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)
llm = ChatDeepSeek(model="deepseek-chat")

# Create the query generation chain
generate_queries = (
    prompt_rag_fusion
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))  # Split the generated queries by line
)
print("✅ Multi-query generator configured")

# Step 5: Test examples
print("\n🎯 Starting RRF reranking test...")
questions = [
    "What are the famous tourist attractions in Shanxi?",
    "What is the historical background of the Yungang Grottoes?",
    "What is the cultural and religious significance of Mount Wutai?"
]

# Perform RRF retrieval and reranking for each question
for idx, question in enumerate(questions, 1):
    print(f"\n{'='*50}")
    print(f"🔍 Question {idx}: {question}")
    print('='*50)

    # Step 1: Generate multiple queries
    print("\n1️⃣ Generating multiple related queries...")
    queries = generate_queries.invoke({"question": question})
    # Filter out empty queries
    queries = [q.strip() for q in queries if q.strip()]
    print(f"Generated {len(queries)} queries:")
    for i, query in enumerate(queries, 1):
        print(f"  Query {i}: {query}")

    # Step 2: Retrieve for each query
    print(f"\n2️⃣ Performing vector retrieval for each query...")
    all_results = []
    for i, query in enumerate(queries, 1):
        print(f"  Retrieving for query {i}: {query}")
        docs = retriever.invoke(query)
        all_results.append(docs)
        print(f"    Retrieved {len(docs)} relevant documents")

    # Step 3: Fuse the results using the RRF algorithm
    print(f"\n3️⃣ Fusing retrieval results using the RRF algorithm...")
    reranked_docs = reciprocal_rank_fusion(all_results)

    # Step 4: Show the final results
    print(f"\n4️⃣ Final RRF reranked results (showing top 3):")
    print(f"Fused {len(reranked_docs)} unique documents in total")

    for i, (doc, score) in enumerate(reranked_docs[:3], 1):
        print(f"\n📄 Rank {i} (RRF score: {score:.4f}):")
        # Truncate to the first 200 characters to avoid overly long output
        content_preview = doc.page_content[:200].replace('\n', ' ').strip()
        print(f"   Content preview: {content_preview}...")

        # Show the document's source info (if available)
        if hasattr(doc, 'metadata') and doc.metadata:
            source = doc.metadata.get('source', 'Unknown source')
            print(f"   Source: {source}")

print(f"\n🎉 RRF reranking test complete!")
print("\n📋 RRF algorithm summary:")
print("- ✅ Multi-angle query generation: understands the user's question from different angles")
print("- ✅ Multi-result fusion: combines the strengths of multiple retrieval results")
print("- ✅ Rank optimization: reorders documents using the RRF algorithm")
print("- ✅ Improved recall: reduces omissions from relying on a single query")
print("- ✅ Improved relevance: documents appearing multiple times receive higher weight")
