from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.llms.deepseek import DeepSeek 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Configure global settings
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
# Prepare the game knowledge text
game_knowledge = """The combat system of "Chronicles of Godslaying: Wukong" is exquisitely designed. Players can freely switch between multiple combat forms during battle, each with its own unique advantages. In Vajra form... while in Demon Buddha form..."""
# Create the Document object
documents = [Document(text=game_knowledge)]
# Create the hierarchical node parser and process the document
# Use HierarchicalNodeParser to create a hierarchical text structure
# chunk_sizes represents the text chunk size at each level
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[256, 128, 64]  # Chunk sizes from root node to leaf node
)
nodes = node_parser.get_nodes_from_documents(documents)
# Get the leaf nodes (smallest granularity text chunks) and root nodes
leaf_nodes = get_leaf_nodes(nodes)
root_nodes = get_root_nodes(nodes)
# Build the storage and index
# Create the document store and add all nodes
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
# Create the storage context
storage_context = StorageContext.from_defaults(docstore=docstore)
# Create the vector index for the leaf nodes
base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context
)
# Create the base retriever and the auto-merging retriever
base_retriever = base_index.as_retriever(similarity_top_k=6)
auto_merging_retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    verbose=True  # Show the merging process
)
# Prepare test questions
test_questions = [
    # "What is the difference between Vajra form and Demon Buddha form in the game?",
    "What are the characteristics of the Bronze Cloud Staff in its different forms?",
    # "How is the game's difficulty designed?"
]
print("=== Results from the auto-merging retriever ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    # Retrieve using the auto-merging retriever
    merge_nodes = auto_merging_retriever.retrieve(question)
    print(f"Retrieved {len(merge_nodes)} merged nodes:")
    for node in merge_nodes:
        print(f"\nSimilarity: {node.score}")
        print(f"Content: {node.node.text}")
        print("-" * 50)
print("\n=== Results from the base retriever (for comparison) ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    # Retrieve using the base retriever
    base_nodes = base_retriever.retrieve(question)
    print(f"Retrieved {len(base_nodes)} base nodes:")
    for node in base_nodes:
        print(f"\nSimilarity: {node.score}")
        print(f"Content: {node.node.text}")
        print("-" * 50)
