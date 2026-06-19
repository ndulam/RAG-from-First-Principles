from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain.chains import RetrievalQA
# System setup documents: focus on specific game mechanics and systems
system_docs = [
    "《Chronicles of Godslaying∙Wukong》uses a unique transformation system as its core combat mechanic",
    "In Vajra form, you can use heavy weapons, increasing attack and defense power",
    "Demon Buddha form focuses on spell attacks, unleashing powerful magical damage",
    "You can switch between different forms at any time during combat to chain combos",
    "Game difficulty is divided into three levels: Normal, Hard, and Asura"
]
# Lore documents: focus on the story and background setup
lore_docs = [
    "The game's setting is a fictional mythological world blending Eastern mythological elements",
    "Wukong reawakens in the game after being sealed for 500 years",
    "The world contains deities and demons from multiple factions such as Buddhism and Taoism",
    "The player, playing as Wukong, must seek the truth among the various factions",
    "Game scenes include mountains and architecture rendered in an ink-wash painting style"
]
# Create two different retrievers: BM25 + vector retriever
from langchain_community.retrievers import BM25Retriever # BM25 retriever
from langchain_community.vectorstores import FAISS # Vector database, not a retriever itself
from langchain.retrievers import EnsembleRetriever # Hybrid retriever
# Create the BM25 retriever
bm25_retriever = BM25Retriever.from_texts(
    system_docs + lore_docs,
    metadatas=[{"source": "system" if i < len(system_docs) else "lore"} 
               for i in range(len(system_docs) + len(lore_docs))]
)
bm25_retriever.k = 2
# Create a vector retriever
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = FAISS.from_texts(
    system_docs + lore_docs,
    embed_model,
    metadatas=[{"source": "system" if i < len(system_docs) else "lore"}
               for i in range(len(system_docs) + len(lore_docs))]
)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Create a vector retriever based on the vector database
# Create the hybrid retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], # Hybrid retriever, containing two retrievers
    weights=[0.5, 0.5] # Weights, used to balance the contribution of the two retrievers -> weighted reranker
)
# Create a QA chain using the hybrid retriever and a QA chain using a single retriever (for comparison)
llm = ChatDeepSeek(model="deepseek-chat")
# Create the hybrid retrieval QA chain -> somewhat like an ensemble learning method
ensemble_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=ensemble_retriever,
    return_source_documents=True
)
# Create a standalone vector retrieval QA chain (for comparison)
vector_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_retriever,
    return_source_documents=True
)
# Test different types of queries
test_queries = [
    "What is the transformation system in the game like?",  # System mechanics query
    "What is the world background of the game like?",      # Background setup query
    "What combat forms does Wukong have?"           # Hybrid query
]
for query in test_queries:
    print(f"\nQuery: {query}")
    print("\n1. Hybrid retrieval results:")
    ensemble_docs = ensemble_retriever.invoke(query)
    print("Retrieved documents:")
    for i, doc in enumerate(ensemble_docs, 1):
        print(f"{i}. [{doc.metadata['source']}] {doc.page_content}")
    print("\n2. Vector retrieval results (for comparison):")
    vector_docs = faiss_retriever.invoke(query)
    print("Retrieved documents:")
    for i, doc in enumerate(vector_docs, 1):
        print(f"{i}. [{doc.metadata['source']}] {doc.page_content}")
#  Test QA performance
print("\n=== QA Performance Test ===")
test_questions = [
    "What are the characteristics of the Vajra form?",
    "What is the distribution of factions in the game like?",
]
for question in test_questions:
    print(f"\nQuestion: {question}")
    print("\n1. Answer using hybrid retrieval:")
    ensemble_result = ensemble_qa.invoke({"query": question})
    print(f"Answer: {ensemble_result['result']}")
    print("\nSource documents used:")
    for i, doc in enumerate(ensemble_result['source_documents'], 1):
        print(f"{i}. [{doc.metadata['source']}] {doc.page_content}")
    print("\n2. Answer using pure vector retrieval (for comparison):")
    vector_result = vector_qa.invoke({"query": question})
    print(f"Answer: {vector_result['result']}")
    print("\nSource documents used:")
    for i, doc in enumerate(vector_result['source_documents'], 1):
        print(f"{i}. [{doc.metadata['source']}] {doc.page_content}")