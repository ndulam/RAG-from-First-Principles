from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
import os

# 1. Load Document
loader = TextLoader("99-EN/black-myth-wukong/black_myth_wukong_setting.txt")
documents = loader.load()

# 2. Split Document
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 3. Create Vector Database
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
db = FAISS.from_documents(texts, embeddings)

# 4. Retrieve Relevant Content
query = "What are the characteristics and combat style of Baigujing?"
docs = db.similarity_search(query)
retrieved_content = docs[0].page_content

# 5. Define Prompt Template
template = """
Based on the following retrieved information:
{context}

Please analyze in detail and generate a character analysis report in the following format:

Character Name: [Provide full name]

Background Story: Introduce the character's origin and background, relationships with other characters, and their role in the story.
Skill Characteristics: Introduce the character's main skills and abilities, description of special abilities, and combat style characteristics.
Combat Strategy: Introduce the character's main attack methods, defense mechanisms, special performance in combat, strengths and weaknesses.

Please conduct a detailed analysis based on the information, ensuring accuracy and coherence.
"""

# Create PromptTemplate and LLM
prompt = PromptTemplate(
    input_variables=["context"],
    template=template
)

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Generate Text
formatted_prompt = prompt.format(context=retrieved_content)
response = llm.invoke(formatted_prompt)
print(response)