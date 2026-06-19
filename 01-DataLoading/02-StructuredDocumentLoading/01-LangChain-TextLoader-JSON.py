from langchain_community.document_loaders import TextLoader
print("=== TextLoader load result ===")
text_loader = TextLoader("../../99-EN/black-myth-wukong/journey_to_the_west_characters.json")
text_documents = text_loader.load()
print(text_documents)
