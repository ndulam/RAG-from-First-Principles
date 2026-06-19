# Read a single txt file
import os
from langchain_community.document_loaders import TextLoader
# Get the directory the current script is in
script_dir = os.path.dirname(__file__)
print(f"Directory of the current script: {script_dir}")
# Build the full path from the relative path
file_dir = os.path.join(script_dir, '../../90-Data/BlackMythWukong/setup.txt')

loader = TextLoader(file_dir)
documents = loader.load()
print(documents)
