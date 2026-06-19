from langchain_community.document_loaders import YoutubeLoader

# Load the document along with its metadata
docs = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=zDvnAY0zH7U", add_video_info=True
).load()

# Inspect the metadata of the first loaded document
print(docs[0].metadata)
