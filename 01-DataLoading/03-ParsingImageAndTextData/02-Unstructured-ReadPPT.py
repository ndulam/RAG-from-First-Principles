"""
Q: When using unstructured to parse PPT and other Office files, what should I do if I get
FileNotFoundError: soffice command was not found?
A: This happens because libreoffice is missing from the system. unstructured needs to call
libreoffice's soffice command to process Office documents.
The fix is to install libreoffice on the system.
On Debian/Ubuntu, you can install it with:
sudo apt-get update && sudo apt-get install -y libreoffice

- Install instructions: https://www.libreoffice.org/get-help/install-howto/
- Mac: https://formulae.brew.sh/cask/libreoffice
- Debian: https://wiki.debian.org/LibreOffice
"""
from unstructured.partition.ppt import partition_ppt
# Parse the PPT file
ppt_elements = partition_ppt(filename="99-EN/black-myth-wukong/black_myth_wukong_slides.pptx")
print("PPT content:")
# for element in ppt_elements:
#     print(element.text)

from langchain_core.documents import Document
# Convert to the Documents data structure
documents = [
Document(page_content=element.text,
  	     metadata={"source": "99-EN/black-myth-wukong/black_myth_wukong_slides.pptx"})
    for element in ppt_elements
]

# Print the converted Documents
print(documents[0:3])
