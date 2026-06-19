import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever # Multi-perspective query retriever
# Load environment variables from the .env file
load_dotenv()

# Configure logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
# Load the game-related documents and build the vector database
loader = TextLoader("../../99-EN/black-myth-wukong/black_myth_wukong_setting.txt", encoding='utf-8')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Chroma.from_documents(documents=splits, embedding= embed_model)
# Use MultiQueryRetriever to generate multi-perspective queries
llm = ChatDeepSeek(model="deepseek-chat", temperature=0, api_key=os.getenv("DEEPSEEK_API_KEY"))
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
query = "Um, so I just started playing this game and it feels really hard. What's the difficulty level like, and how many levels are there? On the Putuo Mountain level, uh, I just can't get past it no matter what. What skill should I learn first? I'm new, please help!"
# Call the MultiQueryRetriever to decompose the query
docs = retriever_from_llm.invoke(query)
print(docs)