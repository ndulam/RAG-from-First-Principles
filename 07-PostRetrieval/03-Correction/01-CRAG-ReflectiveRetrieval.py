"""
CRAG (Corrective Retrieval-Augmented Generation) Reflective Retrieval System

CRAG is an improved RAG approach that boosts retrieval quality through the
following steps:
1. Retrieve: fetch relevant documents from the vector database
2. Grade: assess the relevance of the retrieved documents
3. Decide: based on the grading results, decide whether to generate the
   answer directly or rewrite the query
4. Correct: if the documents are not relevant, rewrite the query and run a
   web search
5. Generate: produce the final answer based on the filtered, relevant
   documents

This approach can automatically detect and correct inaccurate retrieval
results, improving the reliability of the RAG system.
"""

# ================================
# Part 1: Data preparation and vector database construction
# ================================

#1 Build an index for 3 blog posts
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables (including the OpenAI API key, etc.)
load_dotenv()

# Define the blog post URLs to index
# These are technical blog posts about AI agents, prompt engineering, and
# adversarial attacks
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",        # AI agents
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",  # Prompt engineering
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",      # LLM adversarial attacks
]

# Load the content of each URL with WebBaseLoader
docs = [WebBaseLoader(url).load() for url in urls]
# Flatten the nested list into a single list of documents
docs_list = [item for sublist in docs for item in sublist]

# Create a text splitter that uses the tiktoken encoder to accurately count
# tokens
# chunk_size=250: each document chunk has at most 250 tokens
# chunk_overlap=0: no overlap between chunks, avoiding duplicated information
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
# Split the documents into small chunks for easier retrieval and processing
doc_splits = text_splitter.split_documents(docs_list)

# Create the vector database
# Use Chroma as the vector store and OpenAI's embedding model for vectorization
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",  # Collection name
    embedding=OpenAIEmbeddings(),  # Use OpenAI's text-embedding-ada-002 model
)
# Convert the vector store into a retriever for subsequent similarity search
retriever = vectorstore.as_retriever()

# ================================
# Part 2: Retrieval grader - the core component of CRAG
# ================================

#2 Retrieval grader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define the data model for the grading result
# Use Pydantic to ensure a consistent, type-safe output format
class GradeDocuments(BaseModel):
    """Binary score for relevance of a retrieved document.

    This class defines the output format for document relevance grading,
    ensuring the model only returns a clear 'yes' or 'no' judgment.
    """

    binary_score: str = Field(
        description="'yes' if the document is relevant to the question, 'no' otherwise"
    )

# Create a language model with structured output
# temperature=0.5: moderate randomness, balancing consistency and creativity
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
# Constrain the model output to the GradeDocuments format
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Build the grading prompt template
# The system prompt defines the grader's role and grading criteria
system = """You are a grader assessing the relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Build the retrieval grading chain: prompt template + structured language model
retrieval_grader = grade_prompt | structured_llm_grader

# Test the grader
question = "agent memory"  # Test question: about agent memory
docs = retriever.get_relevant_documents(question)  # Retrieve relevant documents
doc_txt = docs[1].page_content  # Get the content of the second document
# Print the grading result to verify the grader works correctly
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

# ================================
# Part 3: RAG generation chain
# ================================

#3 Set up the generation model
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Pull a pre-built RAG prompt template from the LangChain Hub
# This template is specifically designed for answering questions based on context
prompt = hub.pull("rlm/rag-prompt")

# Create the language model used for generating answers
# temperature=0: ensures consistent, reproducible output
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Document formatting function
def format_docs(docs):
    """Format a list of documents into a single string.

    Args:
        docs: a list of document objects

    Returns:
        str: the document contents joined by double newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG generation chain: prompt template + language model + string output parser
rag_chain = prompt | llm | StrOutputParser()

# Test the generation chain
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

# ================================
# Part 4: Query rewriter
# ================================

#4 Set up the question rewriter
# Create the language model used for query rewriting
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

# System prompt for query rewriting
# The goal is to rewrite vague or inaccurate queries into a form better
# suited for search
system = """You are a question re-writer that converts an input question to a better version that is optimized for web search. \n
     Look at the input and try to reason about the underlying semantic intent / meaning."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

# Create the query-rewriting chain
question_rewriter = re_write_prompt | llm | StrOutputParser()
# Test the query-rewriting functionality
question_rewriter.invoke({"question": question})

# ================================
# Part 5: Web search tool
# ================================

#5 Set up the web search tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Create the web search tool
# k=3: return at most 3 search results
web_search_tool = TavilySearchResults(k=3)

# ================================
# Part 6: Graph state definition
# ================================

#6 Set up the imports required for CRAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough

#7 Define the graph state
from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of the CRAG workflow graph.

    This state is passed throughout the entire CRAG pipeline and carries all
    the key information needed during processing.

    Attributes:
        question: the user's original question or the rewritten question
        generation: the final answer generated by the language model
        web_search: a flag indicating whether a web search is needed ("Yes"/"No")
        documents: the list of retrieved documents (original retrieval
            results or web search results)
    """

    question: str        # The question currently being processed
    generation: str      # The generated answer
    web_search: str      # Flag for whether a web search is needed
    documents: List[str] # List of documents

# ================================
# Part 7: CRAG workflow node functions
# ================================

from langchain.schema import Document

def retrieve(state):
    """
    Retrieve node: fetch relevant documents from the vector database

    This is the first step of the CRAG pipeline, retrieving potentially
    relevant documents based on the user's question.

    Args:
        state (dict): the current graph state, must contain the 'question' key

    Returns:
        state (dict): the updated state, with the 'documents' key added
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Use the vector retriever to fetch relevant documents
    # Returns the most relevant document chunks based on semantic similarity
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate node: produce an answer based on the retrieved documents

    This is the final step of the CRAG pipeline, using the filtered,
    relevant documents to generate the final answer.

    Args:
        state (dict): the current graph state, containing question and documents

    Returns:
        state (dict): the updated state with the generation key added
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # Use the RAG chain to generate the answer
    # The documents serve as context, combined with the question to produce a response
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Document grading node: assess the relevance of the retrieved documents

    This is CRAG's core innovation: an LLM evaluates whether each retrieved
    document is genuinely relevant. Only relevant documents are kept, and if
    no relevant documents remain, the state is flagged for a web search.

    Args:
        state (dict): the current graph state

    Returns:
        state (dict): documents updated to the filtered, relevant documents,
            with the web_search flag set
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Initialize the filtering results
    filtered_docs = []           # Stores the relevant documents
    web_search = "No"           # Defaults to not needing a web search
    has_relevant_docs = False   # Flag for whether any relevant documents were found

    # Grade the relevance of each retrieved document
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score

        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)    # Keep the relevant document
            has_relevant_docs = True   # Mark that a relevant document was found
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue  # Skip the irrelevant document

    # CRAG's key logic: only run a web search if there are no relevant
    # documents at all. This avoids unnecessary web searches and improves
    # efficiency.
    if not has_relevant_docs:
        web_search = "Yes"

    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query(state):
    """
    Query transformation node: rewrite the query to improve search quality

    When none of the retrieved documents are relevant, rewrite the original
    query to obtain better search results.

    Args:
        state (dict): the current graph state

    Returns:
        state (dict): the question key updated with the rewritten question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Use the query rewriter to produce an improved question
    # The rewritten question is usually more specific and better suited for search
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": question}

def web_search(state):
    """
    Web search node: fetch supplementary external information

    When the local document store cannot provide relevant information,
    obtain additional information through a web search.

    Args:
        state (dict): the current state, containing
            - question: the question (possibly rewritten)
            - documents: the list of documents

    Returns:
        state (dict): with the web search results appended to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Use the Tavily search tool to run a web search
    search_results = web_search_tool.invoke(question)

    # Format the search results as a Document object
    # This keeps the format consistent with locally retrieved documents
    search_results_str = "\n".join([str(result) for result in search_results])
    web_results = Document(page_content=search_results_str)
    documents.append(web_results)

    return {"documents": documents, "question": question}

# ================================
# Part 8: Conditional edge logic
# ================================

### Edge handling function

def decide_to_generate(state):
    """
    Decision node: determine the next action

    This is the key decision point in the CRAG workflow:
    - If relevant documents were found: generate the answer directly
    - If no relevant documents were found: transform the query and run a web search

    Args:
        state (dict): the current graph state

    Returns:
        str: the name of the next node to execute
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]  # Get the flag for whether a web search is needed
    state["documents"]

    if web_search == "Yes":
        # All local documents were graded as not relevant
        # Need to rewrite the query and run a web search for better information
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # Relevant documents were found, so the answer can be generated directly
        print("---DECISION: GENERATE---")
        return "generate"

# ================================
# Part 9: Build and compile the CRAG workflow graph
# ================================

#8 Compile the graph
from langgraph.graph import END, StateGraph, START

# Create the state graph workflow
workflow = StateGraph(GraphState)

# Add all nodes to the workflow graph
workflow.add_node("retrieve", retrieve)              # Retrieval node
workflow.add_node("grade_documents", grade_documents) # Document grading node
workflow.add_node("generate", generate)              # Answer generation node
workflow.add_node("transform_query", transform_query) # Query transformation node
workflow.add_node("web_search_node", web_search)     # Web search node

# Build the edges connecting the workflow graph
# Define the execution order and conditional jumps between nodes

# 1. From the start node to the retrieve node
workflow.add_edge(START, "retrieve")

# 2. From retrieve to document grading
workflow.add_edge("retrieve", "grade_documents")

# 3. From document grading to a conditional branch
# Choose the next node based on the return value of decide_to_generate
workflow.add_conditional_edges(
    "grade_documents",           # Source node
    decide_to_generate,          # Decision function
    {
        "transform_query": "transform_query",  # If a web search is needed
        "generate": "generate",                # If relevant documents were found
    },
)

# 4. Run a web search after the query transformation
workflow.add_edge("transform_query", "web_search_node")

# 5. Generate the answer after the web search
workflow.add_edge("web_search_node", "generate")

# 6. End after generating the answer
workflow.add_edge("generate", END)

# Compile the workflow graph into an executable app
app = workflow.compile()

# ================================
# Part 10: Run the CRAG system
# ================================

#9 Use the graph to answer a question

from pprint import pprint

# Prepare the input question
# First question: about the types of agent memory (English)
inputs = {"question": "What are the types of agent memory?"}

# Second example question (Chinese, commented out)
# inputs = {"question": "Why is Shanxi province rich in tourism resources?"}

print("=== CRAG Workflow Execution ===")

# Stream the execution of the CRAG workflow
# The stream method lets us observe the execution of each node
for output in app.stream(inputs):
    for key, value in output.items():
        # Print the name of the node currently executing
        pprint(f"Node '{key}':")

        # Optional: print the full state information for each node
        # This is helpful for debugging and understanding the workflow
        # pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

print("=== Final Generation Result ===")
# Print the final generated answer
pprint(value["generation"])

"""
CRAG workflow summary:

1. Retrieve: fetch candidate documents from the vector database
2. Grade (grade_documents): use an LLM to assess document relevance
3. Decide (decide_to_generate): choose a path based on the grading results
4a. Direct path: if relevant documents were found -> generate the answer
4b. Corrective path: if no relevant documents were found -> transform the
    query -> web search -> generate the answer

This design ensures the system can automatically detect and correct
retrieval errors, significantly improving the accuracy and reliability of
the RAG system.
"""