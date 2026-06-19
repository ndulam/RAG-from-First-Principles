from langchain_community.document_loaders import PyPDFLoader
file_path = "../../99-EN/black-myth-wukong/black_myth_wukong_slides.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()
print(f"Loaded {len(pages)} PDF pages")
for page in pages:
    print(page.page_content)
