import os
from dotenv import load_dotenv
load_dotenv()

# 1. Load the document
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Black_Myth:_Wukong",)
)
docs = loader.load()

# 2. Split the document into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. Set up the embedding model
from langchain_openai import OpenAIEmbeddings # pip install langchain-openai
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# 4. Create the vector store
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 5. Build the user query
question = "What game scenes are there in Black Myth: Wukong?"

# 6. Search the vector store for relevant documents and prepare the context
retrieved_docs = vector_store.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 7. Build the prompt template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
                Answer the question based on the context below. If the context
                doesn't contain relevant information, say "I cannot find relevant
                information in the provided context."
                Context: {context}
                Question: {question}
                Answer:"""
                                          )

# 8. Use the large language model to generate the answer
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer.content)


