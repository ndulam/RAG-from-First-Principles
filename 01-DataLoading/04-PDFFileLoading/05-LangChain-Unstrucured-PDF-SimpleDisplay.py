file_path = ("../../99-EN/assets/shanxi-tourism/云冈石窟-en.pdf")
from langchain_unstructured import UnstructuredLoader
loader = UnstructuredLoader(
    file_path=file_path,  # PDF file path
    strategy="hi_res",    # use the high-resolution strategy for document processing
    # partition_via_api=True,  # chunk the document via the API
    # coordinates=True,     # extract text coordinate info
)
docs = []

# lazy_load() is a lazy-loading method
# rather than loading all documents into memory at once, it loads them one at a time as needed
# this can save memory when working with large PDF files
for doc in loader.lazy_load():
    docs.append(doc)

print(docs)
