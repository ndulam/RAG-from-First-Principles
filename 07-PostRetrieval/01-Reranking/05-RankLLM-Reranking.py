from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
import torch

"""
RankLLM Reranking Algorithm Implementation

RankLLM is a reranking method based on large language models (LLMs), leveraging the LLM's
powerful language understanding capabilities to rerank documents.

Core principles:
1. Use the LLM's deep language understanding to judge the relevance between query and documents
2. Guide the LLM's ranking decisions through prompt engineering
3. Combine the LLM's reasoning ability to handle complex semantic relationships

Technical characteristics:
- Deep semantic understanding: based on the LLM's powerful language understanding
- Strong reasoning ability: capable of complex logical reasoning and semantic matching
- High flexibility: can be adapted to different domains and tasks via prompt tuning
- Good explainability: the LLM can provide reasons and explanations for the ranking

Comparison with other methods:
- vs BERT-based models: deeper semantic understanding, capable of more complex reasoning
- vs traditional reranking: able to understand context and implicit information
- vs embedding models: considers not only similarity but also logical relationships

Suitable scenarios:
- Applications requiring extremely high precision
- Queries that require complex reasoning
- Document retrieval in highly specialized domains
- Reranking tasks that require explainability

Notes:
- Relatively high computational cost (calls the LLM API)
- Relatively high latency
- Requires careful prompt design
"""

print("🔄 Initializing the RankLLM reranking system...")

# 1. Document loading and preprocessing
print("📖 Loading and preprocessing the document...")
doc_path = "99-EN/shanxi-tourism/yungang_grottoes.txt"
print(f"Document path: {doc_path}")

print("  🔤 Loading the document with TextLoader...")
documents = TextLoader(doc_path).load()
print(f"  ✅ Document loaded successfully, raw document count: {len(documents)}")

print("  ✂️  Splitting the document...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 500 characters per document chunk
    chunk_overlap=100     # 100-character overlap between chunks to preserve context continuity
)
texts = text_splitter.split_documents(documents)
print(f"  📊 Number of document chunks after splitting: {len(texts)}")

# Add a unique ID to each document chunk
print("  🆔 Adding a unique identifier to each document chunk...")
for idx, text in enumerate(texts):
    text.metadata["id"] = idx
    text.metadata["chunk_size"] = len(text.page_content)
print("  ✅ Document preprocessing complete")

# 2. Create the vector retriever
print(f"\n🔍 Creating the FAISS vector retriever...")
print("  📥 Loading the Chinese embedding model...")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")  # Use an embedding model optimized for Chinese
print("  🏗️  Building the FAISS vector index...")
retriever = FAISS.from_documents(texts, embed_model).as_retriever(
    search_kwargs={"k": 20}  # First stage retrieves the top 20 documents
)
print(f"  ✅ Vector retriever created, will return the top 20 candidate documents")

# 3. GPU memory optimization (if using GPU)
print(f"\n🧹 Optimizing GPU memory usage...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("  🗑️  GPU cache cleared")
else:
    print("  💻 Currently running in CPU mode")

# 4. Configure the RankLLM reranker
print(f"\n🤖 Configuring the RankLLM reranker...")
print("  ⚙️  RankLLM configuration parameters:")
print("    - top_n: 3 (return the top 3 documents in the end)")
print("    - model: gpt (use the GPT model)")
print("    - gpt_model: gpt-4o-mini (an efficient GPT model)")

# Configure OPENAI proxy information
# OPENAI_BASE_URL = "https://vip.apiyi.com/v1"
# OPENAI_API_KEY = ""
# os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

compressor = RankLLMRerank(
    top_n=3,                    # Return the top 3 most relevant documents in the end
    model="gpt",                # Use the GPT model for reranking
    gpt_model="gpt-4o-mini"     # Choose the efficient GPT-4o-mini model
)
print("  ✅ RankLLM reranker configuration complete")

# 5. Create the contextual compression retriever
print(f"\n🔗 Creating the contextual compression retriever...")
print("  📋 Combining the vector retriever with the RankLLM reranker...")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,     # Use RankLLM as the compressor (reranker)
    base_retriever=retriever        # Use FAISS as the base retriever
)
print("  ✅ Retrieval pipeline built: FAISS retrieval → RankLLM reranking")

# 6. Run the query and rerank
print(f"\n🎯 Starting the query and reranking process...")
query = "What are some famous statues at the Yungang Grottoes?"
print(f"Query: {query}")

print(f"\nStage 1 - FAISS vector retrieval:")
print("  🔍 Retrieving candidate documents based on semantic similarity...")

print(f"\nStage 2 - RankLLM reranking:")
print("  🤖 Calling the GPT model for deep semantic reranking...")
print("  ⏳ Processing (LLM inference takes some time)...")

try:
    compressed_docs = compression_retriever.invoke(query)
    print(f"  ✅ RankLLM reranking complete")
    print(f"  📊 Returning {len(compressed_docs)} high-quality documents in the end")

    # 7. Format and print the reranking results
    def pretty_print_docs(docs):
        """
        Pretty-print function for documents

        Purpose: display the reranked document results in an easy-to-read format

        Args:
            docs (list): the list of reranked documents
        """
        print(f"\n{'='*60}")
        print(f"🏆 Final RankLLM reranking results")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Reranked documents (in descending order of relevance):")

        result_parts = []
        for i, doc in enumerate(docs, 1):
            doc_info = f"\n📄 Rank {i}:\n"
            doc_info += f"   Document content:\n{doc.page_content}\n"

            # Show document metadata
            if hasattr(doc, 'metadata') and doc.metadata:
                doc_info += f"   Document ID: {doc.metadata.get('id', 'unknown')}\n"
                doc_info += f"   Content length: {doc.metadata.get('chunk_size', len(doc.page_content))} characters\n"
                if 'source' in doc.metadata:
                    doc_info += f"   Source file: {doc.metadata['source']}\n"

            result_parts.append(doc_info)

        return "\n" + ("-" * 100) + "\n".join(result_parts)

    # Print the formatted results
    formatted_result = pretty_print_docs(compressed_docs)
    print(formatted_result)

except Exception as e:
    print(f"  ❌ RankLLM reranking failed: {str(e)}")
    print("  💡 Possible causes:")
    print("    - GPT API key not configured or invalid")
    print("    - Network connection issue")
    print("    - API quota exhausted")
    print("    - Document content format issue")
    print("  🔧 Suggested checks:")
    print("    - OpenAI API key configuration")
    print("    - Network connection status")
    print("    - Whether the document file exists")

# 8. Resource cleanup
print(f"\n🧹 Cleaning up system resources...")
try:
    # Clean up the RankLLM model (if needed)
    if 'compressor' in locals():
        del compressor
        print("  🗑️  RankLLM model resources released")

    # Clear the GPU cache again
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("  🗑️  GPU cache cleared")

    print("  ✅ Resource cleanup complete")
except Exception as e:
    print(f"  ⚠️  Warning during resource cleanup: {str(e)}")

print(f"\n📋 RankLLM reranking summary:")
print("- ✅ Deep understanding: based on the LLM's powerful language understanding")
print("- ✅ Reasoning ability: capable of complex logical reasoning and semantic matching")
print("- ✅ High precision: leverages state-of-the-art language model technology")
print("- ✅ Explainable: the LLM can provide reasons and grounds for the ranking")
print("- ⚠️  High cost: requires calling the LLM API, relatively expensive")
print("- ⚠️  High latency: LLM inference takes relatively long")
print("- 💡 Best practice: suited to important queries requiring extremely high precision")
print("- 🔧 Optimization tip: design the prompt carefully to improve reranking quality")