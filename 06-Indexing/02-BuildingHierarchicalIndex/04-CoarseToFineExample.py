from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import IndexNode, Document
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from typing import List
# Configure global settings
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
# Create descriptions of game scenes (main documents)
scene_descriptions = [
    """
    Flower-Fruit Mountain: This is the birthplace of the Great Sage Wukong. The mountain is perpetually wreathed in immortal mist, with waterfalls cascading from a thousand meters up,
    forming the "Heavenly River Flying Falls." Various immortal herbs and elixirs grow on the mountain, along with many animals that have cultivated into spirits.
    """,
    """
    Water Curtain Cave: Located at the summit of Flower-Fruit Mountain, with a naturally formed water curtain in front of the cave entrance, serving both as a natural barrier and a sacred place of cultivation.
    """,
    """
    East Sea Dragon Palace: A magnificent palace located at the bottom of the East Sea, decorated with coral and luminous pearls. This is where Wukong obtained the Pillar That Calms the Seas.
    """
]
# Convert scene descriptions into Document objects
documents = [Document(text=desc) for desc in scene_descriptions]
# Use the node parser to convert documents into nodes
doc_nodes = Settings.node_parser.get_nodes_from_documents(documents)
# Create IndexNodes representing the hierarchical relationship
# Create scene detail information (simulating documents with details)
scene_details = [
    """
    Flower-Fruit Mountain detailed setup
    1. Geographic location: within the country of Aolai, in the Eastern Purvavideha continent
    2. Natural environment: exotic flowers and herbs that never wither year-round, clear mountain springs and waterfalls, dense ancient forests
    3. Special areas: the Immortal Fruit Garden, growing various spirit fruits; the Training Ground, a flat and open area for cultivation; the Rest Area, a place for the monkey clan to rest
    """,
    """
    Water Curtain Cave detailed setup
    1. Structure: exterior, a huge natural rock cavern; entrance, a 30-zhang-tall water curtain waterfall; interior, an intricate system of caves
    2. Functional zones: the Training Hall, equipped with all kinds of cultivation equipment; the Treasure Vault, storing various magic treasures and elixirs, protected by powerful warding formations; the Council Hall, able to hold hundreds of the monkey clan, a place for discussing important matters.
    """,
    """
    East Sea Dragon Palace detailed setup
    1. Architectural features: materials, coral, pearls, luminous pearls; scale, spanning dozens of li; style, an undersea palace complex.
    2. Important locations: the Dragon King's Treasury, storing countless treasures such as luminous pearls, and also housing divine artifacts like the Pillar That Calms the Seas; the Armory, holding all kinds of water-element magic implements and divine weapons; the Main Hall, the formal hall for receiving guests, where assemblies of the aquatic races can be held.
    """
]
# Create a corresponding IndexNode for each detail entry, and create the corresponding query engine
index_nodes = []
index_id_query_engine_mapping = {}
for idx, detail_text in enumerate(scene_details):
    # Create the IndexNode, processing the text before putting it into the f-string
    index_id = f"detail{idx}"
    first_line = detail_text.split('\n')[1].strip()
    index_node = IndexNode(text=f"This node contains {first_line}", index_id=index_id)
    index_nodes.append(index_node)
    # Create the corresponding TextNode, and build a separate index and query engine
    detail_node = Document(text=detail_text)
    detail_index = VectorStoreIndex.from_documents([detail_node])
    detail_query_engine = detail_index.as_query_engine()
    # Add the query engine to the mapping
    index_id_query_engine_mapping[index_id] = detail_query_engine
    # Print the current mapping status
    print(f"\nCurrent index ID: {index_id}")
    print(f"Index node text: {index_node.text}")
    print(f"Length of corresponding scene detail: {len(detail_text)} characters")
    print(f"Query engine type: {type(detail_query_engine).__name__}")
print("-" * 30)

# Merge document nodes and index nodes
all_nodes = doc_nodes + index_nodes
# Build the main vector index
vector_index = VectorStoreIndex(all_nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=2)
# Create the RecursiveRetriever object
recursive_retriever = RecursiveRetriever(
    "vector",  # ID of the root retriever
    retriever_dict={"vector": vector_retriever},  # Retriever mapping
    query_engine_dict=index_id_query_engine_mapping,  # Query engine mapping
    verbose=True,  # Enable verbose output
)
# Create the RetrieverQueryEngine, setting the response mode to "compact"
response_synthesizer = get_response_synthesizer(response_mode="compact")
# Create the RetrieverQueryEngine, passing in the RecursiveRetriever and the response synthesizer
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer,
)
# Define the query function
def query_scene(question: str):
    print(f"Question: {question}\n")
    response = query_engine.query(question)
    print(f"Answer: {str(response)}\n")
    print("-" * 50)
# Example queries
if __name__ == "__main__":
    questions = [
        "What is special about Flower-Fruit Mountain?",
        "Describe the internal structure of Water Curtain Cave in detail.",
        "What treasures are stored in the East Sea Dragon Palace?",
    ]

    for q in questions:
        query_scene(q)