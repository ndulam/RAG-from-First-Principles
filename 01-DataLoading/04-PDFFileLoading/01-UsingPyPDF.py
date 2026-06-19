from langchain_community.document_loaders import PyPDFLoader
file_path = "90-Data/BlackMythWukong/black mythWukong.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()
print(f"Loaded {len(pages)} PDF pages")
for page in pages:
    print(page.page_content)
