import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import BaseOutputParser
from langchain.prompts import PromptTemplate
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
# Custom output parser
class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Filter out empty lines
output_parser = LineListOutputParser()
# Custom query prompt template
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an experienced game customer support agent. Please rewrite the user's query from 5 different perspectives to help the player get more detailed game guidance.
                Make sure each query focuses on a different aspect, such as skill selection, combat strategy, or equipment loadout.
                The user's original question: {question}
                Please give 5 different queries, one per line.""",
)
# Set up the LLM processing pipeline
llm = ChatDeepSeek(model="deepseek-chat", temperature=0, api_key=os.getenv("DEEPSEEK_API_KEY"))
llm_chain = QUERY_PROMPT | llm | output_parser
# MultiQueryRetriever using the custom prompt template
retriever = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(),
    llm_chain=llm_chain,
    parser_key="lines"
)
# Run the multi-perspective query
query = "Um, so I just started playing this game and it feels really hard. What's the difficulty level like, and how many levels are there? On the Putuo Mountain level, uh, I just can't get past it no matter what. What skill should I learn first? I'm new, please help!"
# Call the MultiQueryRetriever to decompose the query
docs = retriever.invoke(query)
print(docs)