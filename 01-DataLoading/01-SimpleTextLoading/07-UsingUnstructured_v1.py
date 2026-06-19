from unstructured.partition.text import partition_text
text = "90-Data/BlackMythWukong/setup.txt"
elements = partition_text(text)
for element in elements:
    print(element)

# Use vars() to inspect all available metadata
for i, element in enumerate(elements):
    print(f"\n--- Element {i+1} ---")
    print(f"Type: {type(element)}")
    print(f"Element type: {element.__class__.__name__}")
    print(f"Text content: {element.text}")

    # Show metadata
    if hasattr(element, 'metadata'):
        print("Metadata:")
        metadata = vars(element.metadata)
        valid_metadata = {k: v for k, v in metadata.items()
                         if not k.startswith('_') and v is not None}
        for key, value in valid_metadata.items():
            print(f"  {key}: {value}")
