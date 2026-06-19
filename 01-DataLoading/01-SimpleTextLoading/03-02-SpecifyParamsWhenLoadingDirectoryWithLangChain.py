from langchain_community.document_loaders import DirectoryLoader

import os
# Get the directory the current script is in
script_dir = os.path.dirname(__file__)
print(f"Directory of the current script: {script_dir}")
# Build the full path from the relative path
data_dir = os.path.join(script_dir, '../../90-Data/BlackMythWukong')

loader = DirectoryLoader(data_dir,
                         glob="**/*.md",
                         use_multithreading=True,
                         show_progress=True,
                         )
docs = loader.load()
print(f"Number of documents: {len(docs)}")  # print the total document count
print(docs[0])  # print the first document
