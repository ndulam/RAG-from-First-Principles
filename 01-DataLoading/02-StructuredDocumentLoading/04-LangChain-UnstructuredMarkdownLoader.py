from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

markdown_path = "99-EN/black-myth-wukong/black_myth_wukong_versions.md"
loader = UnstructuredMarkdownLoader(markdown_path)

data = loader.load()
print(data[0].page_content[:250])

loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")
data = loader.load()
print(f"Number of documents: {len(data)}\n")
for document in data:
    print(f"{document}\n")
