from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = TextLoader("99-EN/shanxi-tourism/yungang_grottoes.txt")
documents = loader.load()
# Define the list of separators, used in priority order
separators = ["\n\n", ".", "，", " "] # . is period, ， is comma, " " is space
# Create the recursive splitter, passing in the separator list
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separators=separators
)
chunks = text_splitter.split_documents(documents)
print("\n=== Document Chunking Results ===")
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Document Chunk {i} ---")
    print(f"Content: {chunk.page_content}")
    print(f"Metadata: {chunk.metadata}")
    print("-" * 50)