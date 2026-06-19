# Use WebBaseLoader to load a web page
import bs4
from langchain_community.document_loaders import WebBaseLoader
page_url = "https://en.wikipedia.org/wiki/Black_Myth:_Wukong"
# loader = WebBaseLoader(web_paths=[page_url])
# docs = []
# docs = loader.load()
# assert len(docs) == 1
# doc = docs[0]
# print(f"{doc.metadata}\n")
# print(doc.page_content.strip()[:3000])


# Only parse the main body of the article
loader = WebBaseLoader(
    web_paths=[page_url],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(id="bodyContent"),
    },
    # bs_get_text_kwargs={"separator": " | ", "strip": True},
)
docs = []
docs = loader.load()
assert len(docs) == 1
doc = docs[0]
print(f"{doc.metadata}\n")
print(doc.page_content)
