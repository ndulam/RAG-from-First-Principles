import logging
from langchain.retrievers import RePhraseQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Configure logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)
# Load the game documentation data
loader = TextLoader("90-Data/BlackMythWukong/BlackMythWukongsetup.txt", encoding='utf-8')
data = loader.load()
# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
# Create the vector store
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Chroma.from_documents(documents=all_splits, embedding= embed_model)
# Set up the RePhraseQueryRetriever
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
retriever_from_llm = RePhraseQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm # Use the DeepSeek model as the rephraser
)
# Example input: a game-related query
query = "Um, so I just started playing this game, and it feels really hard. On the Putuo Mountain level, uh, I just can't get past it no matter what. What skill should I learn first? I'm new, please help!"
# Call the RePhraseQueryRetriever to rewrite the query
docs = retriever_from_llm.invoke(query)
print(docs)