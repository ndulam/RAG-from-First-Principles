from langchain_community.document_loaders import DirectoryLoader, TextLoader

import os
# Get the directory the current script is in
script_dir = os.path.dirname(__file__)
print(f"Directory of the current script: {script_dir}")
# Build the full path from the relative path
data_dir = os.path.join(script_dir, '../../90-Data/BlackMythWukong')

# Load every Markdown file in the directory
loader = DirectoryLoader(data_dir,
                         glob="**/*.md",
                         loader_cls=TextLoader # specify the loader to use
                         )
docs = loader.load()
print(docs[0].page_content[:100])  # print the first 100 characters of the first document's content
