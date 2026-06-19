# 1. Load the document
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

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
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

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

llm = ChatOpenAI(
    model="deepseek-reasoner",  # model name supported by the DeepSeek API
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,        # controls the randomness of the output (0-1, higher = more random)
    max_tokens=2048,        # maximum output length
    top_p=0.95,            # controls the diversity of the output (0-1)
    presence_penalty=0.0,   # presence penalty coefficient (-2.0 to 2.0)
    frequency_penalty=0.0,  # frequency penalty coefficient (-2.0 to 2.0)
    api_key=os.getenv("DEEPSEEK_API_KEY")  # load the API key from an environment variable
)
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)


