import textract
text = textract.process("data/BlackMythWukong/black mythWukong.pdf")
print(text)

# Textract vs PyMuPDF / Unstructured:

# Advantages:
# 1. Supports multiple document formats (not just PDF, also doc, odt, etc.)
# 2. Simple to use, unified API
# 3. Automatically detects document encoding
# 4. No extra dependencies required
# 5. Good for quickly extracting text content

# Disadvantages:
# 1. Relatively limited functionality, text extraction only
# 2. Cannot retrieve document structure information
# 3. Does not support image extraction
# 4. Weaker handling of complex layouts
# 5. Cannot retrieve metadata
# 6. Poorer performance on large files
