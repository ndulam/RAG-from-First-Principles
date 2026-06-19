from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
import os

# Example data
examples = [
    {
        "context": "A large manufacturing enterprise's supply chain system experienced delays, leading to a 15% decrease in production efficiency. Investigation revealed that this was mainly due to chaotic supplier management and inaccurate inventory forecasting.",
        "answer": """Problem Analysis Report:
                    Core Issue: Inefficient supply chain
                    Impact: 15% reduction in production efficiency
                    Main Reasons:
                    - Imperfect supplier management system
                    - Insufficient accuracy of inventory forecasting system

                    Proposed Solutions:
                    1. Optimize supplier evaluation system
                    2. Introduce intelligent forecasting system
                    3. Establish real-time monitoring mechanism"""
    },
    {
        "context": "A technology company's employee turnover rate reached 25%, mainly concentrated in the R&D department, affecting product iteration progress.",
        "answer": """Problem Analysis Report:
                    Core Issue: High employee turnover rate
                    Impact: 25% turnover rate
                    Main Reasons:
                    - Insufficient competitiveness of salary and benefits
                    - Limited career development opportunities

                    Proposed Solutions:
                    1. Optimize compensation system
                    2. Improve promotion mechanism
                    3. Enhance working environment"""
    }
]

# Create vector database
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
example_texts = [ex["context"] for ex in examples]
db = FAISS.from_texts(example_texts, embeddings)

# User input problem description
current_issue = """A retail chain's customer complaint rate increased by 40% in the past three months, mainly focusing on delivery timeliness and product quality, affecting brand reputation."""

# Retrieve the most similar example
docs = db.similarity_search(current_issue, k=1)
most_similar_example = next(ex for ex in examples if ex["context"] == docs[0].page_content)

# Construct prompt
prompt = """Here is an example of enterprise problem analysis:

Example:
Based on the following situation:
{example_context}

{example_answer}

Now, based on the following problem, generate an analysis report in the same format:
{input_context}

Please maintain professionalism and actionability in the analysis.
"""

# Create LLM
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Format prompt and generate response
formatted_prompt = prompt.format(
    example_context=most_similar_example["context"],
    example_answer=most_similar_example["answer"],
    input_context=current_issue
)

print(formatted_prompt)

response = llm.invoke(formatted_prompt)
print(response)