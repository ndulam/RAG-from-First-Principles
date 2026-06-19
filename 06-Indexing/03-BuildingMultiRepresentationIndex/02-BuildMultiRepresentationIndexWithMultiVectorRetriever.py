# Load documents
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
# Create document summaries
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | ChatDeepSeek(model="deepseek-chat")
    | StrOutputParser()
)
summaries = chain.batch(docs, {"max_concurrency": 5})
# Set up the multi-vector retriever
from langchain.storage import InMemoryByteStore # In-memory storage
from langchain_huggingface import HuggingFaceEmbeddings # Embedding model
from langchain_community.vectorstores import Chroma # Vector database
from langchain.retrievers.multi_vector import MultiVectorRetriever # Multi-vector retriever
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh") # Embedding model
vectorstore = Chroma(collection_name="summaries", embedding_function= embed_model) # Vector database
store = InMemoryByteStore() # In-memory storage
id_key = "doc_id" # Document ID
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
# Add documents and summaries to the retriever
import uuid
from langchain_core.documents import Document
doc_ids = [str(uuid.uuid4()) for _ in docs]
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
# Use the retriever to query
query = "Memory in agents"
retrieved_docs = retriever.get_relevant_documents(query,n_results=1)

print(retrieved_docs)