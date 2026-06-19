import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from the .env file
load_dotenv()

# Initialize the language model and the embedding model
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
# Prepare the game knowledge text and create a Document object.
from langchain.schema import Document
game_knowledge = """
"Chronicles of Godslaying: Wukong" is an action role-playing game. The game's setting is built around an alternate-history mythological world. Players take on the role of the Great Sage Equal to Heaven, Sun Wukong, embarking on an adventure through a world steeped in Eastern mythology. The game's combat system is highly distinctive, featuring a unique "transformation system." Wukong can switch between different forms during battle. Each form has its own distinctive combat style and skill set. The Vajra form emphasizes power-based strikes, delivering overwhelming destructive force. The Demon Buddha form focuses on spell attacks, capable of unleashing powerful magical damage. The game world is filled with iconic mythological characters; besides the protagonist Sun Wukong, there are gods and demons from Buddhist, Taoist, and other factions. These characters may be Wukong's allies, or they may be formidable adversaries that must be defeated. The equipment system offers a wide range of weapon choices; besides the famous Ruyi Cudgel, Wukong can also wield various divine weapons and treasures. Different weapons have their own distinctive effects, and players need to choose flexibly based on the battle scene. The game's visual presentation has a strongly Eastern aesthetic; the scenes blend ink-wash painting styles, perfectly rendering mountains, rivers, architecture, and other elements. The combat effects combine traditional Chinese cultural elements with the visual impact of modern games. In terms of difficulty design, boss fights are highly challenging, requiring players to precisely time their combat rhythm and skill usage. The game also offers multiple difficulty options to accommodate players of different skill levels.

"""
# Create a Document object
documents = [Document(page_content=game_knowledge)]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Parent text-chunk splitter (larger chunks)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", "；", ",", " ", ""]
)
# Child text-chunk splitter (smaller chunks)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", ",", " ", ""]
)
# Create the parent and child text chunks
parent_docs = parent_splitter.split_documents(documents)
child_docs = child_splitter.split_documents(documents)
# Create the storage and retriever, establishing a two-tier storage system
from langchain.retrievers import ParentDocumentRetriever # Parent document retriever
from langchain.storage import InMemoryStore # In-memory storage
from langchain_community.vectorstores import Chroma # Vector store
vectorstore = Chroma(
    collection_name="game_knowledge",
    embedding_function=embed_model
)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore, # Vector store
    docstore=store, # Document store
    child_splitter=child_splitter, # Child text-chunk splitter
    parent_splitter=parent_splitter, # Parent text-chunk splitter
)
# Add the text chunks
retriever.add_documents(documents)
# Custom prompt template
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
prompt_template = """Answer the question based on the following context. If you cannot find the answer, say "I cannot find relevant information."
Context:
{context}
Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # QA chain type
    retriever=retriever,# Retriever
    return_source_documents=True, # Whether to return source documents
    chain_type_kwargs={"prompt": PROMPT}
)
# Test the system with actual questions
test_questions = [
    "What forms can Wukong transform into in the game?",
    "What is the game's visual style like?",
]
for question in test_questions:
    print(f"\nQuestion: {question}")
    result = qa_chain({"query": question})
    print(f"\nAnswer: {result['result']}")
    print("\nSource documents used:")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"\nRelated document {i}:")
        print(f"Length: {len(doc.page_content)} characters")
        print(f"Content snippet: {doc.page_content[:150]}...")
        print("---")