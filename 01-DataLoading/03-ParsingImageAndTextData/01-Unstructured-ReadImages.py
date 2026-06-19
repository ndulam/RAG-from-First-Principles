from langchain_community.document_loaders import UnstructuredImageLoader
image_path = "90-文档-Data/黑Wukong/黑Wukong英文.jpg"
loader = UnstructuredImageLoader(image_path)

data = loader.load()
print(data)
