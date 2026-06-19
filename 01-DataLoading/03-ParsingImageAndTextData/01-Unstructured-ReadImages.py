from langchain_community.document_loaders import UnstructuredImageLoader
image_path = "99-EN/assets/black-myth-wukong/black_myth_wukong_english.jpg"
loader = UnstructuredImageLoader(image_path)

data = loader.load()
print(data)
