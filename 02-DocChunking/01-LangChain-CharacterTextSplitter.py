from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
loader = TextLoader("90-Data/Shanxi Cultural Tourism/Yungang Grottoes.txt")
documents = loader.load()
# Configure the splitter: chunk size of 50 characters, no overlap
text_splitter = CharacterTextSplitter(
    chunk_size=100,  # each text chunk is 50 characters
    chunk_overlap=10,  # no overlap between chunks
)
chunks = text_splitter.split_documents(documents)
print("\n=== Document Chunking Results ===")
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Document Chunk {i} ---")
    print(f"Content: {chunk.page_content}")
    print(f"Metadata: {chunk.metadata}")
    print("-" * 50)
