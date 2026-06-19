from langchain_core.documents import Document
documents = [
    Document(
        page_content="Wukong is the eldest disciple.",
        metadata={"source": "master_and_disciples.txt"},
    ),
    Document(
        page_content="Bajie is the second disciple.",
        metadata={"source": "master_and_disciples.txt "},
    ),
]
print(documents)
