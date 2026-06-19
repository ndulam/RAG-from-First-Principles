from unstructured.partition.pdf import partition_pdf
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Global settings
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


# Parse the PDF structure and extract text and tables
file_path = "90-Data/ComplexPDF/billionaires_page-1-5.pdf"  # change this to your file path

elements = partition_pdf(
    file_path,
    strategy="hi_res",  # use the high-precision strategy
    infer_table_structure=True,  # infer table structure
)  # parse the PDF document

# Build a mapping from element ID to element
element_map = {element.id: element for element in elements if hasattr(element, 'id')}

for element in elements:
    if element.category == "Table":
        print("\nTable data:")
        print("Table metadata:", vars(element.metadata))  # use vars() to show all metadata attributes
        print("Table content:")
        print(element.text)  # print the table's text content

        # Get and print the parent node info
        parent_id = getattr(element.metadata, 'parent_id', None)
        if parent_id and parent_id in element_map:
            parent_element = element_map[parent_id]
            print("\nParent node info:")
            print(f"Type: {parent_element.category}")
            print(f"Content: {parent_element.text}")
            if hasattr(parent_element, 'metadata'):
                print(f"Parent node metadata: {vars(parent_element.metadata)}")  # also use vars() to show all metadata
        else:
            print(f"Parent node not found (ID: {parent_id})")
        print("-" * 50)

text_elements = [el for el in elements if el.category == "Text"]
table_elements = [el for el in elements if el.category == "Table"]
