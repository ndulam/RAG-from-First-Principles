# Read a single txt file
import os
from langchain_community.document_loaders import TextLoader
# Get the directory the current script is in
script_dir = os.path.dirname(__file__)
print(f"Directory of the current script: {script_dir}")
# Build the full path from the relative path
file_dir = os.path.join(script_dir, '../../99-EN/black-myth-wukong/black_myth_wukong_setting.txt')

loader = TextLoader(file_dir)
documents = loader.load()
print(documents)
