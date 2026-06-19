# Use LangChain to load every document in a directory

"""
# Requires installing Tesseract OCR
### On Ubuntu, run:
sudo apt update
sudo apt install tesseract-ocr -y

### Notes

When loading a directory, DirectoryLoader tries to find a suitable loader for each
file type it encounters. For complex formats like .pptx (PowerPoint), .pdf, and
.jpg, DirectoryLoader generally relies on the unstructured library to handle them.
The unstructured library in turn depends on the nltk (Natural Language Toolkit)
library when extracting and processing text for certain file types. nltk needs to
download some extra data packages (tokenizers, POS taggers, etc.) before it will
work correctly.

If you hit the error zipfile.BadZipFile: File is not a zip file
this happens inside nltk.data.py, which strongly suggests nltk ran into trouble
while trying to load one of its data packages. It tried to open a file as a zip
archive to look for data, but the file wasn't a valid zip file.


Hello!

Based on the error message and file list you provided, the problem occurs when
langchain_community.document_loaders.DirectoryLoader tries to load the .pptx file
in the directory.

Root cause analysis:

When loading a directory, DirectoryLoader tries to find a suitable loader for each
file type it encounters. For complex formats like .pptx (PowerPoint), .pdf, and
.jpg, DirectoryLoader generally relies on the unstructured library to handle them.
The unstructured library in turn depends on the nltk (Natural Language Toolkit)
library when extracting and processing text for certain file types. nltk needs to
download some extra data packages (tokenizers, POS taggers, etc.) before it will
work correctly.
Your error, zipfile.BadZipFile: File is not a zip file, occurs inside nltk.data.py,
which strongly suggests nltk ran into trouble while trying to load one of its data
packages. It tried to open a file as a zip archive to look for data, but the file
wasn't a valid zip file.
Conclusion: the error isn't because your .pptx file itself is corrupted - it's
because when unstructured calls nltk, nltk can't find or properly load the data
package it needs, causing the BadZipFile error. When the directory only contains
.csv files, DirectoryLoader may use a simpler loader that doesn't depend on
unstructured or nltk, so no error occurs.

# Solution:
# The most direct fix is to
# download the NLTK data packages. Run the following code once in your Python environment
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
"""
import os
from langchain_community.document_loaders import DirectoryLoader

# Get the directory the current script is in
script_dir = os.path.dirname(__file__)
print(f"Directory of the current script: {script_dir}")
# Build the full path from the relative path
data_dir = os.path.join(script_dir, '../../99-EN/black-myth-wukong/')

loader = DirectoryLoader(data_dir)
docs = loader.load()
print(f"Number of documents: {len(docs)}")  # print the total document count
print(docs[0])  # print the first document
