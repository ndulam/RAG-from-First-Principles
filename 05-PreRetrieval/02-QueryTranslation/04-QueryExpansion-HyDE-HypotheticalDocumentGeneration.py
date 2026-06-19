import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Load environment variables from the .env file
load_dotenv()

# Load the documents and build the vector database
loader = TextLoader("../../99-EN/black-myth-wukong/black_myth_wukong_wiki.txt", encoding='utf-8')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectordb = Chroma.from_documents(documents=splits, embedding= embed_model)
# HyDE document generation template
template = """Please write a passage of game content relevant to the following question:
Question: {question}
Content:"""
prompt_hyde = ChatPromptTemplate.from_template(template)
# Initialize the model
llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
# Create the chain that generates the hypothetical document
generate_docs_for_retrieval = (
    prompt_hyde | llm | StrOutputParser()
)
# Example question
question = "What are the main skills of the protagonist in Black Myth: Wukong?"
# Generate the hypothetical document
generated_doc = generate_docs_for_retrieval.invoke({"question": question})
print("\n=== Generated hypothetical document ===")
print(generated_doc)
# Initialize the vector store retriever
retriever = vectordb.as_retriever()
# Retrieve relevant documents
retrieval_chain = generate_docs_for_retrieval | retriever
retrieved_docs = retrieval_chain.invoke({"question": question})
print("\n=== Retrieved relevant documents ===")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nDocument {i}:")
    print(doc.page_content)
# Final answer generation template
answer_template = """Answer the question based on the following content:
{context}
Question: {question}
Answer:"""
answer_prompt = ChatPromptTemplate.from_template(answer_template)
# Create the final question-answering chain
final_rag_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)
# Get the final answer
final_answer = final_rag_chain.invoke({"context": retrieved_docs, "question": question})
print("\n=== Final answer ===")
print(final_answer)