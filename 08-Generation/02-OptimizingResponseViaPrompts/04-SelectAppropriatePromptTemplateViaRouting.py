from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Define prompt templates for different business scenarios
templates = {
    "customer_service": """
    You are a professional customer service representative. Please provide solutions based on the following customer feedback and historical cases:
    
    Historical Cases:
    {similar_cases}
    
    Current Customer Feedback: {customer_feedback}
    
    Please provide:
    1. Problem Analysis
    2. Specific Solution
    3. Preventive Measures
    """,
    
    "technical_support": """
    You are a technical expert. Please provide solutions based on the following technical issues and historical cases:
    
    Historical Cases:
    {similar_cases}
    
    Current Issue Description: {issue_description}
    
    Please provide:
    1. Problem Diagnosis
    2. Resolution Steps
    3. Technical Recommendations
    """,
    
    "business_analysis": """
    You are a business analyst. Please provide analysis based on the following business issues and historical cases:
    
    Historical Cases:
    {similar_cases}
    
    Current Issue Description: {business_issue}
    
    Please provide:
    1. Problem Analysis
    2. Impact Assessment
    3. Improvement Suggestions
    """
}

# Example database
case_database = {
    "customer_service": [
        "Customer reported product packaging damage, we provided free replacement service and improved packaging materials",
        "Customer complained about delivery delay, we optimized logistics routes and provided compensation plan",
        "Customer reported product quality issues, we conducted quality inspection and provided refund service"
    ],
    "technical_support": [
        "System crashed frequently, solved by updating server configuration and optimizing code",
        "Database connection timed out, solved by increasing connection pool and optimizing query statements",
        "API response slow, improved performance by adding caching and optimizing algorithms"
    ],
    "business_analysis": [
        "Sales declined, market research found it was due to competitor price war, adjusted marketing strategy",
        "High customer churn rate, customer satisfaction survey found it was a service experience issue, improved service process",
        "Operating costs increased, reduced costs through process optimization and automation"
    ]
}

# Create routing function
def get_prompt_template_by_question(question):
    # Use LLM for intent recognition
    intent_prompt = f"""
    Please analyze which scenario type the following question belongs to:
    Question: {question}
    
    Available Scenarios:
    - customer_service: Customer service related questions
    - technical_support: Technical support related questions  
    - business_analysis: Business analysis related questions

    Only return the corresponding scenario identifier, e.g., 'customer_service'
    """
    
    intent = llm.invoke(intent_prompt).strip()
    
    if intent in templates:
        return PromptTemplate.from_template(templates[intent])
    else:
        raise ValueError(f"Unrecognized scenario type: {question}, identified as: {intent}")

# Get similar cases
def get_similar_cases(scenario, query, k=2):
    # Create vector database
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.from_texts(case_database[scenario], embeddings)
    
    # Retrieve similar cases
    docs = db.similarity_search(query, k=k)
    return "\n".join([f"- {doc.page_content}" for doc in docs])

# Example usage

# Initialize LLM
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Test routing selection for different scenarios
scenarios = ["customer_service", "technical_support", "business_analysis"]
test_queries = {
    "customer_service": "Product packaging damaged, leading to product damage",
    "technical_support": "System crashes frequently, error code 500", 
    "business_analysis": "Sales declined for three consecutive months"
}

# Iterate and test each scenario
for scenario in scenarios:
    query = test_queries[scenario]
    print(f"\n{'='*20} {scenario} {'='*20}")
    print(f"Input Question: {query}")
    
    # Get corresponding prompt template
    prompt_template = get_prompt_template_by_question(query)
    print("\nSelected Prompt Template:")
    print(prompt_template.template)
    
    # Get similar cases
    similar_cases = get_similar_cases(scenario, query)
    print("\nRetrieved Similar Cases:")
    print(similar_cases)
    
    # Set parameters based on variables in the template
    template_vars = {
        "customer_feedback": query,
        "issue_description": query,
        "business_issue": query,
        "similar_cases": similar_cases
    }
    
    # Generate response
    response = llm.invoke(prompt_template.format(**template_vars))
    print("\nGenerated Response:")
    print(response)