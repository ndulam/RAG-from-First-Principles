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
# 创建向量检索器
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = FAISS.from_texts(
    system_docs + lore_docs,
    embed_model,
    metadatas=[{"source": "system" if i < len(system_docs) else "lore"} 
               for i in range(len(system_docs) + len(lore_docs))]
)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 创建向量检索器，基于向量数据库
# 创建混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], # 混合检索器，包含两个检索器
    weights=[0.5, 0.5] # 权重，用于平衡两个检索器的贡献 -> weightedreranker
)
# 创建使用混合检索器的问答链和使用单一检索器的问答链（用于对比）
llm = ChatDeepSeek(model="deepseek-chat")
# 创建混合检索问答链 -> 有点像集成学习方法
ensemble_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=ensemble_retriever,
    return_source_documents=True
)
# 创建单独的向量检索问答链（用于对比）
vector_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_retriever,
    return_source_documents=True
)
# 测试不同类型的查询
test_queries = [
    "游戏中的变身系统是什么样的？",  # 系统机制查询
    "游戏的世界背景是怎样的？",      # 背景setup查询
    "Wukong有哪些战斗形态？"           # 混合查询
]
for query in test_queries:
    print(f"\n查询：{query}")
    print("\n1. 混合检索结果：")
    ensemble_docs = ensemble_retriever.invoke(query)
    print("检索到的文档：")
    for i, doc in enumerate(ensemble_docs, 1):
        print(f"{i}. [{doc.metadata['source']}] {doc.page_content}")    
    print("\n2. 向量检索结果（对比）：")
    vector_docs = faiss_retriever.invoke(query)
    print("检索到的文档：")
    for i, doc in enumerate(vector_docs, 1):
        print(f"{i}. [{doc.metadata['source']}] {doc.page_content}")
#  测试问答效果
print("\n=== 问答效果测试 ===")
test_questions = [
    "金刚形态的特点是什么？",
    "游戏中的势力分布是怎样的？",
]
for question in test_questions:
    print(f"\n问题：{question}")    
    print("\n1. 使用混合检索的回答：")
    ensemble_result = ensemble_qa.invoke({"query": question})
    print(f"回答：{ensemble_result['result']}")
    print("\n使用的源文档：")
    for i, doc in enumerate(ensemble_result['source_documents'], 1):
        print(f"{i}. [{doc.metadata['source']}] {doc.page_content}")    
    print("\n2. 使用纯向量检索的回答（对比）：")
    vector_result = vector_qa.invoke({"query": question})
    print(f"回答：{vector_result['result']}")
    print("\n使用的源文档：")
    for i, doc in enumerate(vector_result['source_documents'], 1):
        print(f"{i}. [{doc.metadata['source']}] {doc.page_content}")
