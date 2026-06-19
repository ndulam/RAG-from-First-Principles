from langchain_community.document_loaders import CSVLoader
# # Part 1: basic CSV loading and printing records
file_path = "99-EN/black-myth-wukong/black_myth_wukong.csv"
# loader = CSVLoader(file_path=file_path)
# data = loader.load()
# print("Example 1: basic CSV loading and printing the first two records")
# for record in data[:2]:
#     print(record)
# print("-" * 80)

# Part 2: skip the CSV header row and use custom column names
# loader = CSVLoader(
#     file_path=file_path,
#     csv_args={
#         "delimiter": ",",
#         "quotechar": '"',
#         "fieldnames": ["Category", "Name", "Description", "PowerLevel"],
#     },
# )
# data = loader.load()

# print("Example 2: skip the header row and use custom column names")
# for record in data[:2]:
#     print(record)
# print("-" * 80)


# # Part 3: specify the "Name" column as the source_column
# loader = CSVLoader(file_path=file_path, source_column="Name")
# data = loader.load()

# print("Example 3: use the 'Name' column as the primary content source")
# for record in data[:2]:
#     print(record)
# print("-" * 80)


# Part 4: use UnstructuredCSVLoader to load the CSV file
from langchain_community.document_loaders import UnstructuredCSVLoader
loader = UnstructuredCSVLoader(file_path=file_path)
data = loader.load()
print("Example 4: load the file with UnstructuredCSVLoader")
print(data)
print("-" * 80)
