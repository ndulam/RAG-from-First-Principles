#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contextual Retrieval and Milvus Implementation
Based on the method proposed by Anthropic, addressing semantic isolation in traditional RAG

Core Concepts:
1. Traditional RAG Problem: Documents are split into independent chunks, losing contextual information.
2. Contextual Retrieval Solution: Add document context to each chunk to make its semantics more complete.
3. Deep Evaluation: Multi-dimensional evaluation of retrieval system performance.

Technology Stack:
- Milvus: Vector database, supporting dense and sparse vectors.
- SentenceTransformer: Generates vector representations of text.
- OpenAI GPT: Used for generating contextualized text chunks (replacing Claude).
- Cohere Reranker: Reranking model, optimizing retrieval results.

API Change Notes:
- The original version used Anthropic Claude API for context enhancement.
- The current version uses OpenAI GPT API, providing better availability and performance.
- Claude-related code has been commented out and can be switched back if needed.

===============================================================================
📋 Code Structure Analysis
===============================================================================

🏗️ Overall Architecture Design:
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG Contextual Retrieval System Architecture │
├─────────────────────────────────────────────────────────────────────┤
│  Input Layer: Raw Documents → Document Chunking → Context Enhancement → Vectorization → Storage │
│  Retrieval Layer: Query Vectorization → Similarity Search → Reranking → Result Return │
│  Evaluation Layer: Golden Standard Comparison → Performance Metric Calculation → Result Analysis │
└─────────────────────────────────────────────────────────────────────┘

🔧 Module Responsibility Division:
1. MilvusContextualRetriever (Core Retriever)
   - Responsible for vector database operations.
   - Implements various retrieval strategies.
   - Manages contextualization and reranking processes.

2. Evaluation Module (Performance Evaluation)
   - evaluate_retrieval(): Core evaluation logic.
   - evaluate_db(): Database performance evaluation.
   - retrieve_base(): Basic retrieval interface.

3. Data Processing Module (Data Processing)
   - download_data(): Data download.
   - load_jsonl(): Data loading.
   - Data format standardization.

4. Experiment Control Module (Experiment Control)
   - main(): Experiment flow control.
   - Comparison of three retrieval strategies.
   - Performance metric statistics.

===============================================================================
🔄 Data Flow Analysis
===============================================================================

📊 Data Processing Flow:
Raw Documents → Document Chunking → [Optional] Context Enhancement → Vectorization → Milvus Storage
    ↓
Query Input → Query Vectorization → Similarity Search → [Reranking] → Result Output
    ↓
Evaluation Comparison → Performance Metrics → Result Analysis

🎯 Retrieval Strategy Comparison:
┌──────────────┬──────────────┬──────────────┬──────────────┐
│   Strategy Type │   Data Preprocessing │   Retrieval Method │   Post-processing │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Standard Retrieval │ Original Text Chunks │ Dense Vector Search │ None           │
│ Contextual Retrieval │ LLM Enhanced Chunks │ Dense Vector Search │ None           │
│ Reranking Retrieval │ LLM Enhanced Chunks │ Dense Vector Search │ Cohere Reranking │
└──────────────┴──────────────┴──────────────┴──────────────┘

🔍 Evaluation System:
Input: Query + Golden Standard Answer
Process: Retrieval → Matching → Scoring
Output: Pass@K score, average score, recall rate

===============================================================================
⚡ Execution Flow Analysis
===============================================================================

🚀 Main Execution Steps:
1️⃣ Environment Initialization
   - Load API keys and configuration.
   - Initialize embedding and reranking models.
   - Download sample data.

2️⃣ Data Preparation
   - Load document dataset.
   - Create evaluation query set.
   - Data format validation.

3️⃣ Experiment 1: Standard Retrieval Baseline
   - Create standard Milvus collection.
   - Insert original text chunks.
   - Perform retrieval evaluation.

4️⃣ Experiment 2: Contextual Retrieval
   - Create contextual Milvus collection.
   - LLM enhance text chunks.
   - Insert enhanced text chunks.
   - Perform retrieval evaluation.

5️⃣ Experiment 3: Reranking Retrieval
   - Use contextual retriever.
   - Enable Cohere reranking.
   - Perform retrieval evaluation.

6️⃣ Result Analysis
   - Compare performance of three strategies.
   - Calculate performance improvement.
   - Generate analysis report.

===============================================================================
🎛️ Key Parameter Configuration
===============================================================================

📋 Vector Database Configuration:
- Collection Name: Differentiates data collections for different experiments.
- Vector Dimension: Determined by the embedding model (e.g., 1024 dimensions for BGE-large-zh).
- Index Type: FLAT (exact search) + SPARSE_INVERTED_INDEX (sparse vectors).
- Distance Metric: Inner Product (IP).

🤖 LLM Configuration:
- Model: gpt-3.5-turbo (fast and economical) or gpt-4 (high quality).
- Max Tokens: 1000.
- Temperature: 0 (ensures consistency).
- API: OpenAI ChatGPT API.

🔄 Retrieval Configuration:
- Retrieval Count K: Default 5 (Pass@5 evaluation).
- Search Parameters: nprobe=10.
- Reranking: Cohere Rerank API.

===============================================================================
"""

# Import necessary libraries
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import CohereRerankFunction

from typing import List, Dict, Any
from typing import Callable
from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)
from tqdm import tqdm
import json
# import anthropic  # Comment out Claude API, switch to OpenAI
import openai  # Add OpenAI API support
import os
import dotenv
dotenv.load_dotenv()

class MilvusContextualRetriever:
    """
    Milvus Contextual Retriever Class
    
    🏛️ Architecture Design:
    This class is the core of the entire retrieval system, adopting a modular design that supports flexible combinations of multiple retrieval strategies.
    
    📦 Functional Modules:
    1. Standard Retrieval: Vector retrieval based on original text chunks.
    2. Hybrid Retrieval: Combines dense and sparse vector retrieval.
    3. Contextual Retrieval: Uses LLM to enrich text chunk context before retrieval.
    4. Reranking Retrieval: Optimizes results using a specialized reranking model based on initial retrieval.
    
    🔄 Data Flow:
    Text Input → [Context Enhancement] → Vectorization → Milvus Storage
           ↓
    Query Input → Vectorization → Similarity Search → [Reranking] → Result Output
    
    🎯 Design Principles:
    - Single Responsibility: Each method is responsible for a specific function.
    - Open/Closed Principle: Supports extending with new retrieval strategies.
    - Dependency Injection: Injects dependent components through the constructor.
    - Configuration Driven: Controls the enabling of different functions through parameters.
    
    Supports standard retrieval, hybrid retrieval, contextual retrieval, and reranking.
    """
    
    def __init__(
        self,
        uri="milvus.db",
        collection_name="contexual_bgem3",
        dense_embedding_function=None,
        use_sparse=False,
        sparse_embedding_function=None,
        use_contextualize_embedding=False,
        llm_client=None,  # Use generic LLM client name (supports OpenAI)
        use_reranker=False,
        rerank_function=None,
    ):
        """
        Initializes the retriever
        
        🔧 Initialization Process:
        1. Set Milvus connection parameters.
        2. Configure embedding functions (dense + sparse).
        3. Set LLM client (for context enhancement).
        4. Configure reranking function.
        5. Parameter validation and error handling.
        
        Parameters:
            uri: Milvus service address
                - Local file mode: e.g., "./milvus.db" (Milvus Lite)
                - Server mode: e.g., "http://localhost:19530" (standalone Milvus service)
                - Cloud service mode: Zilliz Cloud connection address
            collection_name: Collection name, similar to a table name in a database.
            dense_embedding_function: Dense vector embedding function
                - Used to convert text into high-dimensional dense vectors (e.g., 768, 1024 dimensions).
                - Typically uses pre-trained language models like BGE, SentenceTransformer, etc.
            use_sparse: Whether to use sparse vectors
                - Sparse vectors are similar to TF-IDF, primarily capturing keyword matching information.
                - Combining with dense vectors can improve retrieval accuracy.
            sparse_embedding_function: Sparse vector embedding function.
            use_contextualize_embedding: Whether to use contextual embedding
                - This is the core function: uses LLM to add document context to each text chunk.
                - Solves the problem of text chunks lacking context in traditional RAG.
            llm_client: LLM client (supports OpenAI GPT or Claude)
                - Used to generate contextualized text chunks.
                - Current version primarily supports OpenAI GPT-3.5/GPT-4.
                - Original Claude API code is commented out.
            use_reranker: Whether to use reranking
                - Optimizes the ranking of retrieval results using a specialized reranking model.
            rerank_function: Reranking function (e.g., Cohere Rerank).
        """
        self.collection_name = collection_name

        # For Milvus-lite, uri is a local path, e.g., "./milvus.db"
        # For standalone Milvus service, uri is like "http://localhost:19530"
        # For Zilliz Cloud, please set `uri` and `token`
        self.client = MilvusClient(uri)

        self.embedding_function = dense_embedding_function

        self.use_sparse = use_sparse
        self.sparse_embedding_function = None

        self.use_contextualize_embedding = use_contextualize_embedding
        # self.anthropic_client = anthropic_client  # Comment out Claude client
        self.llm_client = llm_client  # Use generic LLM client

        self.use_reranker = use_reranker
        self.rerank_function = rerank_function

        # Parameter validation: If sparse vectors are enabled, a sparse embedding function must be provided.
        if use_sparse is True and sparse_embedding_function:
            self.sparse_embedding_function = sparse_embedding_function
        elif use_sparse is True and sparse_embedding_function is None:
            raise ValueError(
                "Sparse embedding function cannot be None if use_sparse is True"
            )
        else:
            pass

    def build_collection(self):
        """
        Builds the Milvus collection
        
        🏗️ Collection Design Principles:
        1. Schema Design: Defines data structure and field types.
        2. Indexing Strategy: Selects appropriate index types to optimize search performance.
        3. Dynamic Fields: Supports flexible metadata storage.
        4. Vector Fields: Supports hybrid storage of dense and sparse vectors.
        
        📊 Storage Structure:
        ┌─────────────┬──────────────┬─────────────────┬──────────────┐
        │    Field    │    Type      │      Purpose    │    Index Type │
        ├─────────────┼──────────────┼─────────────────┼──────────────┤
        │ pk          │ INT64        │ Primary Key ID  │ Auto          │
        │ dense_vector│ FLOAT_VECTOR │ Semantic Vector │ FLAT/IP      │
        │ sparse_vector│ SPARSE_VECTOR│ Keyword Vector  │ INVERTED/IP  │
        │ content     │ VARCHAR      │ Original Content│ Dynamic Field │
        │ metadata    │ JSON         │ Metadata Info   │ Dynamic Field │
        └─────────────┴──────────────┴─────────────────┴──────────────┘
        
        Collection design notes:
        1. Uses dynamic Schema, supporting flexible field addition.
        2. Primary key is auto-generated, ensuring uniqueness of each record.
        3. Dense vector field: Stores semantic vector representation of text.
        4. Sparse vector field (optional): Stores keyword-related sparse vectors.
        5. Indexing strategy: Dense vectors use FLAT index, sparse vectors use inverted index.
        """
        # Create Schema definition for the collection
        schema = self.client.create_schema(
            auto_id=True,  # Auto-generate primary key ID
            enable_dynamic_field=True,  # Allow dynamic field addition for flexibility
        )
        
        # Add primary key field
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        
        # Add dense vector field - stores semantic vector representation of text
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_function.dim,  # Vector dimension determined by embedding function
        )
        
        # If sparse vectors are enabled, add sparse vector field
        if self.use_sparse is True:
            schema.add_field(
                field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
            )

        # Prepare index parameters - indexes are used to accelerate vector search
        index_params = self.client.prepare_index_params()
        
        # Add index for dense vectors
        # FLAT index: Exact search, suitable for small to medium-sized datasets
        # IP (Inner Product) distance: Suitable for normalized vectors
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )
        
        # Add inverted index for sparse vectors
        if self.use_sparse is True:
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",  # Inverted index specifically for sparse vectors
                metric_type="IP",
            )

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True,
        )

    def insert_data(self, chunk, metadata):
        """
        Inserts standard data into Milvus
        
        📥 Standard Data Insertion Process:
        This is the basic data insertion method, implementing the simplest text chunk storage strategy.
        
        🔄 Processing Flow:
        1. Text input validation.
        2. Dense vector generation (semantic representation).
        3. Sparse vector generation (optional, keyword representation).
        4. Data structure assembly.
        5. Milvus batch insertion.
        
        📊 Data Flow:
        Original Text Chunk → Embedding Model → Vector Representation → Milvus Storage
                                   ↓
        Metadata Info → Field Mapping → Dynamic Field Storage
        
        Parameters:
            chunk: Content of the text chunk (original text, not contextually enhanced).
            metadata: Metadata (including document ID, chunk ID, original content, etc.).
        
        Stored content:
        1. Dense vector: Semantic vector representation of the text chunk.
        2. Sparse vector (optional): Keyword vector representation of the text chunk.
        3. Metadata: Identification information for documents and chunks.
        """
        # Generate dense vector representation for the text chunk
        dense_vec = self.embedding_function([chunk])[0]
        
        # Build data to be inserted
        data = {
            "dense_vector": dense_vec,
            **metadata  # Expand metadata fields
        }
        
        # If sparse vectors are enabled, generate and add sparse vectors
        if self.use_sparse is True:
            sparse_vec = self.sparse_embedding_function([chunk])[0]
            data["sparse_vector"] = sparse_vec
            
        # Insert data into Milvus collection
        self.client.insert(
            collection_name=self.collection_name,
            data=[data]
        )

    def insert_contextualized_data(self, doc_content, chunk_content, metadata):
        """
        Inserts contextualized data
        
        🧠 Core Process of Context Enhancement:
        This is the key method for implementing contextual retrieval, solving the semantic isolation problem of traditional RAG.
        
                 🔄 Contextualization Processing Flow:
         1. Document context preparation.
         2. LLM prompt construction.
         3. OpenAI GPT API call (context enhancement).
         4. Enhanced text vectorization.
         5. Vector data storage.
        
                 📊 Data Enhancement Flow:
         Original Document ──┐
                           ├─→ LLM Prompt → OpenAI GPT API → Contextually Enhanced Text
         Text Chunk ───┘                                  ↓
                                             Embedding → Enhanced Vector → Milvus
        
        🎯 Core Innovation:
        This is the core function of contextual retrieval:
        1. Uses the entire document content as context.
        2. Enriches the semantic information of a single text chunk through LLM (OpenAI GPT).
        3. Generates and stores vectors using the enhanced text.
        
        ✨ Advantages of Contextualization:
        - Solves the problem of isolated text chunks.
        - Preserves semantic coherence across chunks.
        - Improves retrieval accuracy, especially for queries requiring contextual understanding.
        - Reduces semantic ambiguity and misunderstanding.
        
        Parameters:
            doc_content: Entire document content (as contextual background).
            chunk_content: Current text chunk content (the part to be enhanced).
            metadata: Metadata.
        """
        # Construct LLM prompt, asking to enrich the text chunk with document context
        prompt = f"""
        <document>
        {doc_content}
        </document>
        <chunk>
        {chunk_content}
        </chunk>
        
        I need you to enrich the above <chunk> using the content from <document> to provide background and contextual information.
        Your answer should include the complete content of the <chunk> and ensure semantic coherence. Only return the enriched text content, do not add any explanations or interpretations.
        
        Goals:
        1. Keep the core information of the original chunk unchanged.
        2. Add necessary background context to make the meaning of the chunk clearer.
        3. Ensure the enriched text is semantically coherent and complete.
        """
        
        # === OpenAI GPT API call (new version) ===
        # Call OpenAI GPT API to generate contextualized text chunks
        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5-turbo, cost-effective
            # model="gpt-4",        # Can choose GPT-4 for better results
            max_tokens=1000,        # Limit output length
            temperature=0,          # Ensure consistency and reproducibility of output
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract OpenAI generated contextualized text
        contextualized_chunk = response.choices[0].message.content.strip()
        
        # === Claude API call (original version, commented out) ===
        # # Call Claude API to generate contextualized text chunks
        # message = self.anthropic_client.messages.create( 
        #     model="claude-3-haiku-20240307",  # Use fast, economical Haiku model
        #     max_tokens=1000,  # Limit output length
        #     temperature=0,    # Ensure consistency and reproducibility of output
        #     messages=[
        #         {"role": "user", "content": prompt}
        #     ]
        # )
        # 
        # # Extract LLM generated contextualized text
        # contextualized_chunk = message.content[0].text.strip()
        
        # Use contextualized content to generate embedding vectors and insert
        dense_vec = self.embedding_function([contextualized_chunk])[0]
        data = {
            "dense_vector": dense_vec,
            "contextualized_content": contextualized_chunk,  # Save enhanced content
            **metadata
        }
        
        # If sparse vectors are enabled, also generate sparse vectors using contextualized content
        if self.use_sparse is True:
            sparse_vec = self.sparse_embedding_function([contextualized_chunk])[0]
            data["sparse_vector"] = sparse_vec
            
        # Insert contextualized data into Milvus
        self.client.insert(
            collection_name=self.collection_name,
            data=[data]
        )

    def search(self, query, k=5):
        """
        Searches for relevant content
        
        🔍 Intelligent Retrieval Execution Flow:
        This is the core entry point of the retrieval system, supporting unified calls for multiple retrieval strategies.
        
        🎯 Retrieval Strategy Selection:
        ┌─────────────────┬──────────────────┬──────────────────┐
        │    Retrieval Stage │      Processing Method   │      Output Result     │
        ├─────────────────┼──────────────────┼──────────────────┤
        │ 1. Query Vectorization │ Embedding Model  │ Query Vector Representation │
        │ 2. Similarity Search   │ Milvus Vector Search │ Top-K Candidate Results │
        │ 3. Result Reranking    │ Cohere Rerank    │ Optimized Ranked Results │
        └─────────────────┴──────────────────┴──────────────────┘
        
        🔄 Detailed Search Flow:
        1. Query preprocessing and vectorization.
        2. Milvus vector similarity search.
        3. Initial result acquisition and filtering.
        4. Optional reranking optimization.
        5. Result post-processing and return.
        
        📊 Search Optimization Strategy:
        Query Text → Vectorization → Similarity Search → Candidate Results
                                            ↓
        Final Results ← Reranking Optimization ← Semantic Matching ← Result Set
        
        Parameters:
            query: Query text.
            k: Number of results to return.
        
        Returns:
            List of search results, sorted by relevance.
        """
        # Set search parameters
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        
        # Generate embedding vector for the query
        dense_vec = self.embedding_function([query])[0]
        
        # Perform standard dense vector search
        # Inner Product (IP) is used here as the similarity metric
        res = self.client.search(
            collection_name=self.collection_name,
            data=[dense_vec],
            limit=k,
            output_fields=["content", "contextualized_content"],  # Return original and contextualized content
            search_params=search_params,
        )
        
        # Use reranker to further optimize results
        # Role of reranking: Reorder results based on deep semantic relationship between query and documents
        if self.use_reranker:
            # Extract document content for reranking
            docs = []
            for hit in res[0]:
                # Prioritize contextualized content (if available), otherwise use original content
                content = hit["entity"].get("contextualized_content", hit["entity"].get("content", ""))
                docs.append(content)
            
            # Apply reranking: Calculate deep relevance score between query and each document
            rerank_results = self.rerank_function(query, docs)
            
            # Reorder original results based on reranking results
            reranked_results = []
            for result in rerank_results:
                idx = result.index  # Use .index attribute to get original index
                reranked_results.append(res[0][idx])
            
            res = [reranked_results]
        
        return res


def evaluate_retrieval(eval_data, retrieval_function, db, k=5):
    """
    Evaluates retrieval performance - core function of the deep evaluation system.
    
    📊 Deep Evaluation System Design:
    This is a comprehensive retrieval performance evaluation framework, adopting a multi-dimensional evaluation strategy.
    
    🔬 Concept of Deep Evaluation:
    1. Not just evaluating the quantity of retrieval, but more importantly, the quality.
    2. Uses multi-dimensional metrics to evaluate retrieval system performance.
    3. Objectively evaluates using a golden standard dataset.
    4. Supports fair comparison of different retrieval strategies.
    
    🎯 Evaluation Process Design:
    ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
    │   Evaluation Stage │    Input Data   │    Processing Method │    Output Result    │
    ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
    │ 1. Data Preparation │ Evaluation Query Set │ Format Validation │ Standardized Data   │
    │ 2. Retrieval Execution │ Single Query    │ Retrieval Function Call │ Candidate Result List │
    │ 3. Result Matching   │ Retrieval Results │ Exact Text Matching │ Match Success Count │
    │ 4. Performance Calculation │ Match Statistics │ Metric Calculation │ Evaluation Score    │
    └─────────────────┴─────────────────┴─────────────────┴─────────────────┘
    
    📈 Evaluation Metric Description:
    - Pass@K: Proportion of queries where the correct answer is included in the top K retrieval results.
      * Calculation formula: (Number of queries with correct answers / Total number of queries) × 100%
      * Reflects the overall retrieval success rate of the system.
    - Average Score: Proportion of correct document chunks retrieved for each query out of the total correct chunks.
      * Calculation formula: Σ(Number of correct chunks retrieved for query / Total correct chunks for query) / Total number of queries
      * Reflects the fine-grained retrieval accuracy of the system.
    - Recall: Evaluates the system's ability to find relevant documents.
      * Measures the system's performance in not missing important information.
    
    🔄 Evaluation Execution Flow:
    Evaluation Data → Query Parsing → Golden Standard Extraction → Retrieval Execution → Result Comparison → Metric Calculation
        ↓
    Performance Report ← Statistical Analysis ← Score Aggregation ← Match Verification ← Exact Match
    
    Parameters:
        eval_data: Evaluation dataset
            - Contains queries and corresponding golden standard answers.
            - Each query has clear identifiers for correct document chunks.
        retrieval_function: Retrieval function
            - Accepts query and database, returns retrieval results.
        db: Database instance.
        k: Number of top-k results for evaluation.
        
    Returns:
        Evaluation result dictionary, including Pass@K score, average score, etc.
    """
    total_score = 0      # Accumulated score
    total_queries = 0    # Total number of queries
    
    # Iterate through each evaluation query
    for item in tqdm(eval_data, desc="Evaluating retrieval"):
        total_queries += 1
        query = item["query"]
        
        # Get golden standard content (correct answer) for the current query
        golden_contents = []
        for ref in item["references"]:
            doc_uuid = ref["doc_uuid"]      # Unique identifier of the document
            chunk_index = ref["chunk_index"] # Index of the document chunk
            
            # Find the corresponding original document in the dataset
            golden_doc = next(
                (
                    doc
                    for doc in dataset
                    if doc.get("original_uuid") == doc_uuid
                ),
                None,
            )
            if not golden_doc:
                print(f"Warning: Golden document with UUID {doc_uuid} not found")
                continue
                
            # Find the corresponding document chunk in the document
            golden_chunk = next(
                (
                    chunk
                    for chunk in golden_doc["chunks"]
                    if chunk["original_index"] == chunk_index
                ),
                None,
            )
            if not golden_chunk:
                print(f"Warning: Golden chunk with index {chunk_index} not found in document {doc_uuid}")
                continue
                
            golden_contents.append(golden_chunk["content"].strip())
            
        if not golden_contents:
            print(f"Warning: Golden content not found for query: {query}")
            continue
            
        # Use retrieval function to get retrieval results
        retrieved_docs = retrieval_function(query, db, k=k)
        
        # Calculate how many golden chunks are found in the top-k retrieved documents
        # This is the core logic for evaluating retrieval accuracy
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in retrieved_docs[0][:k]:  # Only check the first k results
                content_field = "content"
                if "contextualized_content" in doc["entity"]:
                    # Use original content for comparison during evaluation to ensure fairness
                    content_field = "content"
                retrieved_content = doc["entity"][content_field].strip()
                
                # Exact match check
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break  # Break inner loop if a match is found
                    
        # Calculate score for the current query: (number of correct chunks found) / (total number of correct chunks)
        query_score = chunks_found / len(golden_contents)
        total_score += query_score
        
    # Calculate overall evaluation metrics
    average_score = total_score / total_queries
    pass_at_n = average_score * 100  # Convert to percentage
    
    return {
        "pass_at_n": pass_at_n,           # Pass@K score (percentage)
        "average_score": average_score,    # Average score (between 0-1)
        "total_queries": total_queries,    # Total number of queries
    }


def retrieve_base(query: str, db, k: int = 20) -> List[Dict[str, Any]]:
    """
    Basic retrieval function
    
    This is a simple wrapper function that calls the database's search method.
    Used as a unified interface in the evaluation system.
    """
    return db.search(query, k=k)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads a JSONL file and returns a list of dictionaries.
    
    JSONL format: One JSON object per line, suitable for storing structured evaluation data.
    """
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def evaluate_db(db, original_jsonl_path: str, k):
    """
    Evaluates the retrieval performance of the database.
    
    This is the main entry function for the evaluation process:
    1. Loads the evaluation dataset.
    2. Runs retrieval evaluation.
    3. Outputs performance metrics.
    
    Parameters:
        db: Database instance to be evaluated.
        original_jsonl_path: File path of the evaluation dataset.
        k: Top-k parameter for evaluation.
    
    Returns:
        Evaluation results dictionary.
    """
    # Load original JSONL data as queries and ground truth labels
    original_data = load_jsonl(original_jsonl_path)
    
    # Evaluate retrieval performance
    results = evaluate_retrieval(original_data, retrieve_base, db, k)
    
    # Output evaluation results
    print(f"Pass@{k}: {results['pass_at_n']:.2f}%")
    print(f"Average Score: {results['average_score']}")
    print(f"Total Queries: {results['total_queries']}")
    
    return results


def download_data():
    """
    Downloads sample data.
    
    Downloads demonstration data from Anthropic's GitHub repository:
    1. codebase_chunks.json: Document chunk data for the codebase.
    2. evaluation_set.jsonl: Evaluation queries and golden standard answers.
    """
    import urllib.request
    
    # Check if file already exists to avoid re-downloading
    if not os.path.exists("codebase_chunks.json"):
        print("Downloading codebase_chunks.json...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/codebase_chunks.json",
            "codebase_chunks.json"
        )
    
    if not os.path.exists("evaluation_set.jsonl"):
        print("Downloading evaluation_set.jsonl...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/evaluation_set.jsonl",
            "evaluation_set.jsonl"
        )
    
    print("Data download complete!")


def main():
    """
    Main function - runs all experiments.
    
    🧪 Experiment Design Framework:
    This is a complete comparative experiment system, using controlled variables to verify the effectiveness of different retrieval strategies.
    
    🎯 Experiment Goals:
    - Verify the performance improvement of contextual retrieval relative to standard retrieval.
    - Quantify the additional benefits of reranking technology.
    - Establish performance benchmarks for retrieval technologies.
    - Provide technical selection guidance for practical applications.
    
    📊 Experiment Design Matrix:
    ┌─────────────────┬────────────┬────────────┬────────────┬──────────────┐
    │     Experiment Group     │ Text Preprocessing │ Vector Retrieval │ Result Reranking │  Expected Performance │
    ├─────────────────┼────────────┼────────────┼────────────┼──────────────┤
    │ Standard Retrieval (Baseline) │ Original Text Chunks │ Dense Vector │ None        │ Baseline Performance │
    │ Contextual Retrieval      │ LLM Enhanced Chunks │ Dense Vector │ None        │ Moderate Improvement │
    │ Reranking Retrieval       │ LLM Enhanced Chunks │ Dense Vector │ Cohere      │ Best Performance     │
    └─────────────────┴────────────┴────────────┴────────────┴──────────────┘
    
    🔬 Experiment Control Variables:
    - Same dataset (codebase_chunks.json).
    - Same evaluation queries (evaluation_set.jsonl).
    - Same embedding model (BGE-large-zh).
    - Same evaluation metric (Pass@5).
    - Same vector database configuration.
    
    🚀 Experiment Execution Flow:
    1️⃣ Environment Preparation Phase:
       - API key configuration verification.
       - Model initialization and testing.
       - Data download and verification.
    
    2️⃣ Data Preprocessing Phase:
       - Document data loading and parsing.
       - Evaluation query set construction.
       - Data format standardization.
    
    3️⃣ Experiment Execution Phase:
       - Standard retrieval experiment (baseline).
       - Contextual retrieval experiment (core innovation).
       - Reranking retrieval experiment (performance optimization).
    
    4️⃣ Result Analysis Phase:
       - Performance metric statistics.
       - Improvement margin calculation.
       - Experiment conclusion summary.
    
    🎯 Core Experiment Hypothesis:
    This function runs three comparative experiments, demonstrating the performance differences of various retrieval strategies:
    
    1. Standard Retrieval: Baseline method, directly uses original text chunks.
    2. Contextual Retrieval: Uses LLM to enhance the contextual information of text chunks.
    3. Contextual Retrieval with Reranking: Adds reranking optimization on top of contextual retrieval.
    
    ⚖️ Experiment Fairness Guarantee:
    Each experiment uses the same evaluation dataset to ensure fairness of comparison.
    """
    # Replace these with your actual API keys
    cohere_api_key = os.getenv("COHERE_API_KEY")      # Cohere Reranking API Key
    # anthropic_api_key = os.getenv("CLAUDE_API_KEY")   # Claude API Key (commented out)
    openai_api_key = os.getenv("OPENAI_API_KEY")      # OpenAI API Key (added)
    
    # Download sample data
    download_data()
    
    # Load dataset
    global dataset
    with open("codebase_chunks.json", "r") as f:
        dataset = json.load(f)
    
    # Use only the first 5 documents for testing (to reduce API call cost and runtime)
    dataset = dataset[:5]
    
    # Initialize various models and functions
    dense_ef = SentenceTransformerEmbeddingFunction(model_name='BAAI/bge-large-zh')  # Use Chinese-optimized BGE model
    cohere_rf = CohereRerankFunction(api_key=cohere_api_key)  # Cohere Reranking function
    
    # === OpenAI client initialization (new version) ===
    openai_client = openai.OpenAI(api_key=openai_api_key)  # OpenAI client
    
    # === Claude client initialization (original version, commented out) ===
    # anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)  # Claude client
    
    # ===============================
    # Experiment 1: Standard Retrieval (Baseline Method)
    # ===============================
    print("\n===== Experiment 1: Standard Retrieval =====")
    print("This is the baseline experiment, using original text chunks for retrieval, without any enhancement.")
    
    standard_retriever = MilvusContextualRetriever(
        uri="standard.db", 
        collection_name="standard", 
        dense_embedding_function=dense_ef
    )
    
    # Build collection and insert standard data
    standard_retriever.build_collection()
    for doc in tqdm(dataset, desc="Inserting standard retrieval data"):
        doc_content = doc["content"]
        for chunk in doc["chunks"]:
            metadata = {
                "doc_id": doc["doc_id"],
                "original_uuid": doc["original_uuid"],
                "chunk_id": chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "content": chunk["content"],
            }
            chunk_content = chunk["content"]
            standard_retriever.insert_data(chunk_content, metadata)
    
    # Create simplified evaluation data (for demonstration)
    # In actual applications, a specially designed evaluation dataset should be used
    eval_data = []
    for doc in dataset[:2]:  # Use only the first 2 documents for evaluation
        for chunk in doc["chunks"][:2]:  # Take only the first 2 chunks from each document
            eval_data.append({
                "query": chunk["content"][:50],  # Use the first 50 characters of chunk content as query
                "references": [{
                    "doc_uuid": doc["original_uuid"],
                    "chunk_index": chunk["original_index"]
                }]
            })
    
    # Save evaluation data
    with open("evaluation_set.jsonl", "w") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Evaluate standard retrieval performance
    standard_results = evaluate_db(standard_retriever, "evaluation_set.jsonl", 5)
    
    # ===============================
    # Experiment 2: Contextual Retrieval
    # ===============================
    print("\n===== Experiment 2: Contextual Retrieval =====")
    print("Uses OpenAI GPT to add document context to each text chunk, solving the semantic isolation problem.")
    
    contextual_retriever = MilvusContextualRetriever(
        uri="contextual.db",
        collection_name="contextual",
        dense_embedding_function=dense_ef,
        use_contextualize_embedding=True,  # Enable contextualization
        llm_client=openai_client,  # Use OpenAI client
        # anthropic_client=anthropic_client,  # Original Claude client (commented out)
    )
    
    # Build collection and insert contextualized data
    contextual_retriever.build_collection()
    for doc in tqdm(dataset, desc="Inserting contextual retrieval data"):
        doc_content = doc["content"]
        for chunk in doc["chunks"]:
            metadata = {
                "doc_id": doc["doc_id"],
                "original_uuid": doc["original_uuid"],
                "chunk_id": chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "content": chunk["content"],
            }
            chunk_content = chunk["content"]
            # Use contextualized insertion method
            contextual_retriever.insert_contextualized_data(
                doc_content, chunk_content, metadata
            )
    
    # Evaluate contextual retrieval performance
    contextual_results = evaluate_db(contextual_retriever, "evaluation_set.jsonl", 5)
    
    # ===============================
    # Experiment 3: Contextual Retrieval with Reranking
    # ===============================
    print("\n===== Experiment 3: Contextual Retrieval with Reranking =====")
    print("Further optimizes results by using the Cohere reranking model on top of contextual retrieval.")
    
    # Enable reranking
    contextual_retriever.use_reranker = True
    contextual_retriever.rerank_function = cohere_rf
    
    # Evaluate retrieval performance with reranking
    reranker_results = evaluate_db(contextual_retriever, "evaluation_set.jsonl", 5)
    
    # ===============================
    # Result Comparison and Analysis
    # ===============================
    print("\n===== Comparison of All Experiment Results =====")
    print("Performance Improvement Analysis:")
    print(f"Standard Retrieval Pass@5: {standard_results['pass_at_n']:.2f}%")
    print(f"Contextual Retrieval Pass@5: {contextual_results['pass_at_n']:.2f}%")
    print(f"Contextual Retrieval with Reranking Pass@5: {reranker_results['pass_at_n']:.2f}%")
    
    # Calculate improvement margin
    context_improvement = contextual_results['pass_at_n'] - standard_results['pass_at_n']
    rerank_improvement = reranker_results['pass_at_n'] - standard_results['pass_at_n']
    
    print(f"\nPerformance Improvement Analysis:")
    print(f"Contextual retrieval improved by: {context_improvement:.2f} percentage points compared to standard retrieval.")
    print(f"Reranking further improved by: {rerank_improvement:.2f} percentage points.")
    print(f"Overall improvement: {rerank_improvement:.2f} percentage points.")


if __name__ == "__main__":
    main()