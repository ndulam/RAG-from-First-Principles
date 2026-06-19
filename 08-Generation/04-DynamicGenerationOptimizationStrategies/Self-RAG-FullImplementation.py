from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()   

# Define the list of URLs to load
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vector database
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

### Retrieval Grader ###

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Data model
class GradeDocuments(BaseModel):
    """Binary score model for grading the relevance of retrieved documents"""

    binary_score: str = Field(
        description="Whether the document is relevant to the question, 'yes' or 'no'"
    )

# Configure LLM and function calling
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader that evaluates the relevance of retrieved documents to a user question.\n 
    This does not need to be a strict test. The goal is to filter out irrelevant retrieval results.\n
    If the document contains keywords or semantic meaning related to the user question, grade it as relevant.\n
    Give a binary score of 'yes' or 'no' to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Retrieval grader and simple test
retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

### Generator ###

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM configuration
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build RAG chain
rag_chain = prompt | llm | StrOutputParser()

# Run
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

### Hallucination Grader ###

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination in generated answers"""

    binary_score: str = Field(
        description="Whether the answer is based on facts, 'yes' or 'no'"
    )

# LLM configuration
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader that evaluates whether the LLM generation is based on retrieved facts.\n 
     Give a binary score of 'yes' or 'no'. 'yes' means the answer is based on/supported by facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
hallucination_grader.invoke({"documents": docs, "generation": generation})

### Answer Grader ###

# Data model
class GradeAnswer(BaseModel):
    """Binary score for evaluating whether the answer solves the problem"""

    binary_score: str = Field(
        description="Whether the answer solves the problem, 'yes' or 'no'"
    )

# LLM configuration
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader that evaluates whether the answer solves/answers the question.\n 
     Give a binary score of 'yes' or 'no'. 'yes' means the answer solves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke({"question": question, "generation": generation})

### Question Rewriter ###

# LLM configuration
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Prompt
system = """You are a question rewriter that converts the input question into a better version more suitable for vector store retrieval.\n 
     Review the input and try to understand the underlying semantic intent/meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Please formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})

from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question: The question
        generation: LLM generated content
        documents: List of documents
    """

    question: str
    generation: str
    documents: List[str]

### Node Definitions ###

def retrieve(state):
    """
    Retrieves documents

    Parameters:
        state (dict): Current graph state

    Returns:
        state (dict): New state with retrieved documents
    """
    print("---Retrieving Documents---")
    question = state["question"]

    # Use the new invoke method
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generates an answer

    Parameters:
        state (dict): Current graph state

    Returns:
        state (dict): New state with LLM generated content
    """
    print("---Generating Answer---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines if retrieved documents are relevant to the question.

    Parameters:
        state (dict): Current graph state

    Returns:
        state (dict): Updated list of documents, containing only relevant ones
    """
    print("---Checking Document Relevance to Question---")
    question = state["question"]
    documents = state["documents"]

    # Grade each document
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---Grade: Document Relevant---")
            filtered_docs.append(d)
        else:
            print("---Grade: Document Irrelevant---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    """
    Transforms the query to produce a better question.

    Parameters:
        state (dict): Current graph state

    Returns:
        state (dict): Updated question
    """
    print("---Transforming Query---")
    question = state["question"]
    documents = state["documents"]

    # Rewrite question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

### Edge Definitions ###

def decide_to_generate(state):
    """
    Decides whether to generate an answer or regenerate the question.

    Parameters:
        state (dict): Current graph state

    Returns:
        str: Binary decision for the next node
    """
    print("---Evaluating Graded Documents---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered for relevance
        # We will regenerate a new query
        print("---Decision: All documents irrelevant to question, transforming query---")
        return "transform_query"
    else:
        # We have relevant documents, so generate an answer
        print("---Decision: Generating Answer---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines if the generated content is based on documents and answers the question.

    Parameters:
        state (dict): Current graph state

    Returns:
        str: Decision for the next node
    """
    print("---Checking for Hallucinations---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check for hallucinations
    if grade == "yes":
        print("---Decision: Generated content based on documents---")
        # Check Q&A
        print("---Evaluating Generated Content vs. Question---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---Decision: Generated content answers the question---")
            return "useful"
        else:
            print("---Decision: Generated content does not answer the question---")
            return "not useful"
    else:
        print("---Decision: Generated content not based on documents, retrying---")
        return "not supported"

from langgraph.graph import END, StateGraph, START

# Create workflow graph
workflow = StateGraph(GraphState)

# Define nodes
workflow.add_node("retrieve", retrieve)  # Retrieve
workflow.add_node("grade_documents", grade_documents)  # Grade documents
workflow.add_node("generate", generate)  # Generate
workflow.add_node("transform_query", transform_query)  # Transform query

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()
# try:
#     # First get PNG binary data
#     png_data = app.get_graph(xray=True).draw_mermaid_png()

#     # Save binary data to graph.png in the current directory
#     with open("08-Response Generation-Generation/04-Dynamic Generation Optimization Strategies/graph.png", "wb") as f:
#         f.write(png_data)

#     print("Saved as: graph.png")
# except Exception as e:
#     print(f"Error saving image: {e}")

# from pprint import pprint

# # Run example 1
# inputs = {"question": "Explain how different types of agent memory work?"}
# for output in app.stream(inputs):
#     for key, value in output.items():
#         # Node
#         pprint(f"Node '{key}':")
#     pprint("\n---\n")

# # Final generation
# pprint(value["generation"])

# # Run example 2
# inputs = {"question": "Explain how chain-of-thought prompting works?"}
# for output in app.stream(inputs):
#     for key, value in output.items():
#         # Node
#         pprint(f"Node '{key}':")
#     pprint("\n---\n")

# # Final generation
# pprint(value["generation"])