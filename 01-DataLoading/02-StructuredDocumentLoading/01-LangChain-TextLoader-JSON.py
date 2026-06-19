from langchain_community.document_loaders import TextLoader
print("=== TextLoader load result ===")
text_loader = TextLoader("90-Data/Chronicles of Godslaying/CharactersAndRoles.json")
text_documents = text_loader.load()
print(text_documents)
