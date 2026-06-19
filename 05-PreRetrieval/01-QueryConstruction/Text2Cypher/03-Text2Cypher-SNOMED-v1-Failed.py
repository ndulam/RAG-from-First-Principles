# Prepare the Neo4j database connection
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

# Neo4j connection configuration
uri = "bolt://localhost:7687"  # Default Neo4j Bolt port
username = "neo4j"
password = os.getenv("NEO4J_PASSWORD")  # Get the password from an environment variable

# Initialize the Neo4j driver
driver = GraphDatabase.driver(uri, auth=(username, password))

# Prepare the SNOMED CT schema description
schema_description = """
You are accessing a SNOMED CT graph database, which mainly contains the following nodes and relationships:

Node types:
1. Concept (concept node)
   - conceptId: unique concept identifier
   - fullySpecifiedName: fully specified concept name
   - preferredTerm: preferred term
   - active: whether active
   - effectiveTime: effective time
   - moduleId: module ID

2. Description (description node)
   - descriptionId: unique description identifier
   - term: term text
   - typeId: description type ID
   - languageCode: language code
   - active: whether active

3. Relationship (relationship node)
   - relationshipId: unique relationship identifier
   - typeId: relationship type ID
   - active: whether active

Relationship types:
1. IS_A: represents a hierarchical relationship between concepts
2. HAS_DESCRIPTION: relationship between a concept and its description
3. HAS_RELATIONSHIP: other relationships between concepts
"""

# Initialize the DeepSeek client
from openai import OpenAI
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# Set the query
user_query = "Find all concepts related to 'Diabetes' and their descriptions"

# Prepare the prompt for generating Cypher
prompt = f"""
Below is a description of the SNOMED CT graph database structure:
{schema_description}
The user's natural language question is as follows:
"{user_query}"
Please note:
1. Use a MATCH clause to match nodes and relationships
2. Use a WHERE clause to filter conditions
3. Use a RETURN clause to specify the returned results
4. Return only the Cypher query statement, without any other explanation, comments, or formatting markers (such as ```cypher)
"""

# Call the LLM to generate the Cypher statement
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a Cypher query expert. Return only the Cypher query statement, without any Markdown formatting or other explanation."},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

# Clean up the Cypher statement, removing any Markdown markers
cypher = response.choices[0].message.content.strip()
cypher = cypher.replace('```cypher', '').replace('```', '').strip()
print(f"\nGenerated Cypher query:\n{cypher}")

# Execute the Cypher query and get the results
def run_query(tx, query):
    result = tx.run(query)
    return [record for record in result]

with driver.session() as session:
    results = session.execute_read(run_query, cypher)
    print(f"Query results: {results}")

# Generate a natural language description
if results:
    nl_prompt = f"""
The query results are as follows:
{results}
Please convert this data into a natural language description that is easy to understand.
The original question was: {user_query}

Requirements:
1. Use clear, easy-to-understand language
2. Include all the queried data information
3. If there are technical terms, explain them appropriately
"""
    response_nl = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a medical information expert responsible for converting SNOMED CT query results into an easy-to-understand natural language description."},
            {"role": "user", "content": nl_prompt}
        ],
        temperature=0.7
    )
    description = response_nl.choices[0].message.content.strip()
    print(f"Natural language description:\n{description}")
else:
    print("No relevant data found.")

# Close the database connection
driver.close()