from llama_index.core import SimpleDirectoryReader
# Use SimpleDirectoryReader to load files in a directory
dir_reader = SimpleDirectoryReader("90-Data/BlackMythWukong")
documents = dir_reader.load_data()
# Check the number and content of the loaded documents
print(f"Number of documents: {len(documents)}")
print(documents[0].text[:100])  # print the first 100 characters of the first document

# Load only one specific file
dir_reader = SimpleDirectoryReader(input_files=["90-Data/BlackMythWukong/setup.txt"])
documents = dir_reader.load_data()
print(f"Number of documents: {len(documents)}")
print(documents[0].text[:100])  # print the first 100 characters of the first document
