from unstructured.documents.elements import Title, NarrativeText, Text
from unstructured.partition.pdf import partition_pdf

file_path = '90-Data/Shanxi Cultural Tourism/Yungang Grottoes-en.pdf'

# Use unstructured to read the PDF directly
elements = partition_pdf(
    filename=file_path,
    strategy="hi_res",
    # include_metadata=True,  # if position info is needed
)

print(elements[0].to_dict())

# Add some debug info to inspect the first element in full
if elements:
    first_elem = elements[0]
    print("=== Details of the first element ===")
    print(f"Type: {type(first_elem)}")
    print(f"Text: {first_elem.text}")
    print("Metadata attributes:")
    print(vars(first_elem.metadata))  # print all metadata attributes
    print("All attributes of the element:")
    print(vars(first_elem))  # print all attributes of the element
    print("="*50)

# Only filter the elements on page 1
page_number = 1
page_elements = [elem for elem in elements if getattr(elem.metadata, "page_number", None) == page_number]

# Iterate over and print details for each element
for i, elem in enumerate(page_elements, 1):
    print(f"\nElement {i}:")
    print(f"  Content: {elem.text}")
    print(f"  Category: {type(elem).__name__}")
    print(f"  ID: {getattr(elem, '_element_id', None)}")
    print("="*50)

# Only filter the Titles on page 1
title_dict = {}

# Collect Titles and build a parent_id -> Title mapping
for elem in elements:
    if (isinstance(elem, Title) and
        getattr(elem.metadata, "page_number", None) == page_number):
        title_id = getattr(elem, '_element_id', None) # get the element's ID; a standalone id also works
        title_text = elem.text.strip()
        if title_text not in [data["title"] for data in title_dict.values()]:
            title_dict[title_id] = {"title": title_text, "content": []}

# Associate each Title with its corresponding Text
for elem in elements:
    if (isinstance(elem, (NarrativeText, Text)) and
        getattr(elem.metadata, "page_number", None) == page_number):
        parent_id = getattr(elem.metadata, "parent_id", None)
        if parent_id in title_dict:
            content = elem.text.strip()
            if content:  # only add non-empty content
                title_dict[parent_id]["content"].append(content)

# Print to the command line
for title_data in title_dict.values():
    if title_data["content"]:  # only print titles that have content
        print("\n=== " + title_data["title"] + " ===")
        for content in title_data["content"]:
            print(content)
        print()
