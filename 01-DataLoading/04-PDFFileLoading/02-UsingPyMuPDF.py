import pymupdf
# Open the PDF file
doc = pymupdf.open("../../99-EN/black-myth-wukong/black_myth_wukong_slides.pdf")
text = [page.get_text() for page in doc]
print(text)

# Example: basic PyMuPDF features
print("=== PyMuPDF basic info extraction ===")
print(f"Number of pages: {len(doc)}")
print(f"Document title: {doc.metadata['title']}")
print(f"Document author: {doc.metadata['author']}")
print(f"Document metadata: {doc.metadata}")  # provides more metadata than Unstructured

# Iterate over every page
for page_num, page in enumerate(doc):
    # Extract text
    text = page.get_text()
    print(f"\n--- Page {page_num + 1} ---")
    print("Text content:", text[:200])  # show the first 200 characters

    # Extract images
    images = page.get_images()
    print(f"Number of images: {len(images)}")

    # Get page links
    links = page.get_links()
    print(f"Number of links: {len(links)}")

    # Get page size
    width, height = page.rect.width, page.rect.height
    print(f"Page size: {width} x {height}")

doc.close()

# PyMuPDF (fitz) vs Unstructured:
# Advantages:
# 1. Faster processing speed
# 2. Finer-grained control over the PDF
# 3. Can retrieve more metadata and document structure information
# 4. Lower memory usage
# 5. No dependency on external tools

# Disadvantages:
# 1. Text extraction is less "intelligent"
# 2. No automatic document structure understanding
# 3. Layout analysis has to be handled manually
