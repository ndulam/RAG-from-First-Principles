from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.postprocessor import PrevNextNodePostprocessor, AutoPrevNextNodePostprocessor
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Configure global settings
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
Settings.node_parser = SentenceSplitter()
# Prepare the game story text
game_story = """When Wukong first awoke, he found himself trapped in an ancient cave. With his memory hazy, all he could recall was that he was the Great Sage Equal to Heaven, Sun Wukong, but he couldn't remember why he was here. There was a broken mirror in the cave, and through it he saw himself covered in scars, with his once-mighty Ruyi Cudgel reduced to a broken stub. After leaving the cave, Wukong encountered a mysterious old monk. The old monk told him that this place was the "Illusory Realm," a special space between reality and illusion. 500 years ago, Heaven suffered an unprecedented calamity—the gods fell, and the celestial realm collapsed. Wukong, who at the time was wreaking havoc in Heaven, was swept up in the catastrophe, losing most of his power and memory, and was sealed within this world. The old monk suggested that Wukong search for the fragments of memory scattered throughout the Illusory Realm. The first stop was the Temple of Forgetting River to the east, where a Mirror of Memory was enshrined that might help him recover some of his memories. However, the temple had been occupied by a group of evil demons, and Wukong needed to defeat them first. At the temple, Wukong used the Mirror of Memory to glimpse parts of the catastrophe that befell Heaven. It turned out that a mysterious ancient force had been manipulating events behind the scenes, harnessing the power of the "Will of All Beings" to distort the laws of heaven and earth. Even though Wukong had been powerful at the time, he had been unable to prevent the disaster. After recovering these memories, the old monk told Wukong that his next stop should be the Mountain of Karmic Fire to the west. There, a clan of transformed demons held more of the truth. But the mountain was perpetually surrounded by raging flames, making it nearly impossible for ordinary beings to approach. Wukong needed to first find the legendary Samadhi Fire Armor in order to enter safely. While searching for the Samadhi Fire Armor, Wukong ran into an old friend, the Demon King. The Demon King told him that after Heaven's collapse, the order of the Six Realms had fallen into chaos, with various factions rising up. Some claimed to be rebuilding Heaven, while others sought to establish an entirely new order. An even greater calamity was brewing. After obtaining the Samadhi Fire Armor, Wukong successfully infiltrated the Mountain of Karmic Fire. In his showdown with the demon clan's leader, he finally recalled more of the truth. It turned out that the ancient force's goal was not simple destruction, but rather to reshape the rules of the entire world. They believed the existing order had fundamental flaws that caused suffering for all beings. Returning to the old monk, Wukong said he wanted to rally forces from all sides to oppose the mastermind behind it all. But the old monk told him that things might not be as simple as they appeared. Whether the world's order should be reshaped was a question with no clear-cut answer. He advised Wukong to keep searching for more of the truth before making a decision. Wukong decided to set out for the Sea of Sunken Stars to the south. Legend had it that an ancient library there held countless texts about the origins of the world. However, before he could depart, the Illusory Realm suddenly shook violently, as if some great upheaval was about to occur..."""
# Create a Document object
documents = [Document(text=game_story)]
# Build the document store and index, parsing the documents using the node_parser in Settings
nodes = Settings.node_parser.get_nodes_from_documents(documents)
# Create the document store and add the nodes
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
# Create the storage context
storage_context = StorageContext.from_defaults(docstore=docstore)
# Build the vector index
index = VectorStoreIndex(nodes, storage_context=storage_context)
# Create different query engines
# Base query engine
base_engine = index.as_query_engine(
    similarity_top_k=1,
    response_mode="tree_summarize"
)
# Query engine with a fixed surrounding context
prev_next_engine = index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[
        PrevNextNodePostprocessor(docstore=docstore, num_nodes=2)
    ],
    response_mode="tree_summarize"
)
# Query engine with automatic surrounding context
auto_engine = index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[
        AutoPrevNextNodePostprocessor(
            docstore=docstore,
            num_nodes=3,
            verbose=True
        )
    ],
    response_mode="tree_summarize"
)
# Test different types of questions against different query engines
test_questions = [
    "What happened after Wukong recovered his memories at the Temple of Forgetting River?",  # Should look forward
    "How did Wukong reach the Mountain of Karmic Fire?",  # Should look backward
    "Why did Wukong wake up in the cave?",  # Should look backward
]
print("=== Results from the base query engine ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    response = base_engine.query(question)
    print(f"Answer: {response}\n")
    print("-" * 50)
print("\n=== Results from the fixed surrounding-context query engine ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    response = prev_next_engine.query(question)
    print(f"Answer: {response}\n")
    print("-" * 50)
print("\n=== Results from the automatic surrounding-context query engine ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    response = auto_engine.query(question)
    print(f"Answer: {response}\n")
print("-" * 50)