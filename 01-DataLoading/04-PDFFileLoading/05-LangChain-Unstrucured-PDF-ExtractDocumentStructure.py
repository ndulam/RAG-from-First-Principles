file_path = ("../../99-EN/assets/shanxi-tourism/云冈石窟-en.pdf")
from langchain_unstructured import UnstructuredLoader
loader = UnstructuredLoader(
    file_path=file_path,
    strategy="hi_res",
    # partition_via_api=True,
    # coordinates=True,
)
docs = []
for doc in loader.lazy_load():
    docs.append(doc)

def extract_basic_structure(docs):
    """Basic structured extraction: organize content by document type"""
    # Define the category mapping
    category_map = {
        'Title': 'title',
        'NarrativeText': 'text',
        'Image': 'image',
        'Table': 'table',
        'Footer': 'footer',
        'Header': 'header'
    }

    # Initialize the structure dict
    structure = {cat: [] for cat in category_map.values()}
    structure['metadata'] = [] # add an extra metadata category

    # Walk through the documents and classify them
    for doc in docs:
        category = doc.metadata.get('category', 'Unknown')
        content = {
            'content': doc.page_content,
            'page': doc.metadata.get('page_number'),
            'coordinates': doc.metadata.get('coordinates')
        }

        target_category = category_map.get(category)
        if target_category:
            structure[target_category].append(content)

    return structure

# Usage example
structure = extract_basic_structure(docs)

# Print the titles on page 2
print("Page 2 titles:")
for title in [t for t in structure['title'] if t['page'] == 2]:
    print(f"- {title['content']}")


def analyze_layout(docs):
    """Analyze the document's page layout structure"""
    layout_analysis = {}

    for doc in docs:
        page = doc.metadata.get('page_number')
        coords = doc.metadata.get('coordinates', {})

        # Initialize the page info
        if page not in layout_analysis:
            layout_analysis[page] = {
                'elements': [],
                'dimensions': {
                    'width': coords.get('layout_width', 0),
                    'height': coords.get('layout_height', 0)
                }
            }

        # Get the element's position info
        points = coords.get('points', [])
        if points:
            # We only need the top-left and bottom-right coordinates
            (x1, y1), (_, _), (x2, y2), _ = points

            # Build the element info
            element = {
                'type': doc.metadata.get('category'),
                'content': (doc.page_content[:50] + '...') if len(doc.page_content) > 50 else doc.page_content,
                'position': {
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'width': x2 - x1,
                    'height': y2 - y1
                }
            }
            layout_analysis[page]['elements'].append(element)

    return layout_analysis

# Usage example
layout = analyze_layout(docs)

# Analyze the layout of page 1
print("Page 1 layout analysis:")
if 1 in layout:
    page = layout[1]
    print(f"Page size: {page['dimensions']['width']} x {page['dimensions']['height']}")
    print("\nElement distribution:")

    # Sort and print elements by vertical position
    for elem in sorted(page['elements'], key=lambda x: x['position']['y1']):
        print(f"\nType: {elem['type']}")
        print(f"Position: ({elem['position']['x1']:.0f}, {elem['position']['y1']:.0f})")
        print(f"Size: {elem['position']['width']:.0f} x {elem['position']['height']:.0f}")
        print(f"Content: {elem['content']}")

# Find parent-child relationships
cave6_docs = []
parent_id = -1
for doc in docs:
    if doc.metadata["category"] == "Title" and "Cave 6" in doc.page_content:
        parent_id = doc.metadata["element_id"]
    if doc.metadata.get("parent_id") == parent_id:
        cave6_docs.append(doc)

for doc in cave6_docs:
    print(doc.page_content)

external_docs = [] # create a list to store child documents of external links
parent_id = -1 # initialize the parent ID to -1
for doc in docs:
    # Check whether the document is a Title type and its content contains "External links"
    if doc.metadata["category"] == "Title" and "External links" in doc.page_content:
        parent_id = doc.metadata["element_id"]
        external_docs.append(doc)
    # Check whether the document's parent_id matches the title ID we found
    if doc.metadata.get("parent_id") == parent_id:
        external_docs.append(doc) # add every child document belonging to this title to the results list
for doc in external_docs:
    print(doc.page_content)


# def find_tables_and_titles(docs):
#   results = []
#   for doc in docs:
#     # Check whether the document is a Table type
#     if doc.metadata.get("category") == "Table":
#       table = doc
#       parent_id = doc.metadata.get("parent_id")
#       # Find the title document corresponding to the table (parent_id matches element_id)
#       title = next((doc for doc in docs if doc.metadata.get("element_id") == parent_id), None)
#       if title:
#         results.append({"table": table.page_content, "title": title.page_content})
#   return results

# results = find_tables_and_titles(cave6_docs)
# if results:
#   for result in results:
#     print("Found table and title:")
#     print(f"Title: {result['title']}")
#     print(f"Table: {result['table']}")
# else:
#   print("No tables and titles found")
