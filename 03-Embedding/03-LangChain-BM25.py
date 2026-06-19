import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma # pip install chromadb
from langchain_community.retrievers import BM25Retriever # pip install rank_bm25

battle_logs = [
    "Wukong wears chainmail armor.",
    "Wukong encountered a demon in the Valley of No Return; the demon attacked, and Wukong blocked with the Bronze Cloud Staff.",
    "Wukong used Flaming Fist to repel the demon, then activated Vajra Body to block the divine weapon's attack.",
    "The demon used Frost Arrow to attack Wukong, but was counterattacked and crushed by Flaming Fist.",
    "Wukong summoned Flaming Fist and Devastating Roar to defeat the demon, then collected the demon's essence."
]
request = "What equipment and moves does Wukong have?"

bm25_retriever = BM25Retriever.from_texts(battle_logs)
bm25_response = bm25_retriever.invoke(request)
print(f"BM25 retrieval results:\n{bm25_response}")

docs = [Document(page_content=log) for log in battle_logs]
load_dotenv()
chroma_vs = Chroma.from_documents(
    docs,
    embedding=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("O3_API_KEY"),
        base_url=os.getenv("O3_BASE_URL")
        )
    )
chroma_retriever = chroma_vs.as_retriever()
chroma_response = chroma_retriever.invoke(request)
print(f"Chroma retrieval results:\n{chroma_response}")

# hybrid_response = list({doc.page_content for doc in bm25_response}) # missing the chainmail armor
# hybrid_response = list({doc.page_content for doc in chroma_response}) # missing the Bronze Cloud Staff
hybrid_response = list({doc.page_content for doc in bm25_response + chroma_response})
print(f"Hybrid retrieval results:\n{hybrid_response}")
prompt = ChatPromptTemplate.from_template("""
                Answer the question based on the following context. If the context
                does not contain relevant information, say "I cannot find relevant
                information in the provided context".
                Context: {context}
                Question: {question}
                Answer:"""
                                          )
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("O3_API_KEY"),
    base_url=os.getenv("O3_BASE_URL"))
doc_content = "\n\n".join(hybrid_response)
answer = llm.invoke(prompt.format(question=request, context=doc_content))
print(f"LLM answer:\n{answer.content}")