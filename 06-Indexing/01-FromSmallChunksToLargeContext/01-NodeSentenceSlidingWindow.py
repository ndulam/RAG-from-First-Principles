import os
from dotenv import load_dotenv
from llama_index.core import  VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import  SentenceWindowNodeParser, SentenceSplitter
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor # Metadata replacement postprocessor

# Load environment variables from the .env file
load_dotenv()

# Configure global settings
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
Settings.text_splitter = SentenceSplitter(separator="\n",  chunk_size=50, chunk_overlap=0)
# Prepare the knowledge text and create a Document object
game_knowledge = """
"Chronicles of Godslaying: Wukong" is an action role-playing game. The game's setting is built around an alternate-history mythological world.
Players take on the role of the Great Sage Equal to Heaven, Sun Wukong, embarking on an adventure through a world steeped in Eastern mythology.
The game's combat system is highly distinctive, featuring a unique "transformation system." Wukong can switch between different forms during battle.
Each form has its own distinctive combat style and skill set. The Vajra form emphasizes power-based strikes, delivering overwhelming destructive force.
The Demon Buddha form focuses on spell attacks, capable of unleashing powerful magical damage.
The game world is filled with iconic mythological characters; besides the protagonist Sun Wukong, there are gods and demons from Buddhist, Taoist, and other factions.
These characters may be Wukong's allies, or they may be formidable adversaries that must be defeated.
The equipment system offers a wide range of weapon choices; besides the famous Ruyi Cudgel, Wukong can also wield various divine weapons and treasures.
Different weapons have their own distinctive effects, and players need to choose flexibly based on the battle scene.
The game's visual presentation has a strongly Eastern aesthetic; the scenes blend ink-wash painting styles, perfectly rendering mountains, rivers, architecture, and other elements.
The combat effects combine traditional Chinese cultural elements with the visual impact of modern games.
In terms of difficulty design, boss fights are highly challenging, requiring players to precisely time their combat rhythm and skill usage.
The game also offers multiple difficulty options to accommodate players of different skill levels."""
# Create a Document object
documents = [Document(text=game_knowledge)]
# Create a sentence parser with a context window (keeps n sentences on each side of the target sentence as context)
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)
# Process the documents with the window parser
nodes = node_parser.get_nodes_from_documents(documents)
# Process the documents with the base parser (for comparison)
base_nodes = Settings.text_splitter.get_nodes_from_documents(documents)
# Build two indexes for comparison
sentence_index = VectorStoreIndex(nodes)
base_index = VectorStoreIndex(base_nodes)
# Create a query engine with the context window
window_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ]
)
# Create the base query engine
base_query_engine = base_index.as_query_engine(
    similarity_top_k=6
)
# Test Q&A
test_questions = [
    "What forms can Wukong transform into in the game?",
    # "What is the game's visual style like?",
    # "How is the game's difficulty designed?"
]
print("=== Retrieval results using the window parser ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    window_response = window_query_engine.query(question)
    print(f"Answer: {window_response}")

    # Show the retrieved original sentence and window content
    print("\nRetrieval details:")
    for node in window_response.source_nodes:
        print(f"Original sentence: {node.node.metadata['original_text']}")
        print(f"Context window: {node.node.metadata['window']}")
        print("---")
print("\n=== Retrieval results using the base parser (for comparison) ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    base_response = base_query_engine.query(question)
print(f"Answer: {base_response}")