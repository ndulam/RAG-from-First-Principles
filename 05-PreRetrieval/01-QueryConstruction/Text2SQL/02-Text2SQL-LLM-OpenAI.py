# Prepare the database connection
import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

conn = sqlite3.connect('data/tourism.db')
cursor = conn.cursor()

# Prepare the schema description
schema_description = """
You are accessing a database that contains two tables:
1. scenic_spots (scenic spot information table)
   - scenic_id (INT): primary key, unique scenic spot identifier
   - scenic_name (VARCHAR): scenic spot name
   - city (VARCHAR): the city it is located in
   - level (VARCHAR): scenic spot rating
   - monthly_visitors (INT): monthly visitor count
2. city_info (city information table)
   - city_id (INT): primary key, unique city identifier
   - city_name (VARCHAR): city name
   - annual_tourism_income (INT): annual tourism income (in yuan)
   - famous_dish (VARCHAR): local specialty dish/snack
"""

# Initialize the OpenAI client
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set the query
user_query = "Find the AAAAA-rated scenic spots in Taiyuan and their monthly visitor counts"

# Prepare the prompt for generating SQL
prompt = f"""
Below is a description of the database structure:
{schema_description}
The user's natural language question is as follows:
"{user_query}"
Please note:
1. The city field in the scenic_spots table stores the city name, corresponding to city_name in the city_info table
2. The two tables should be joined using city_name and city
3. Return only the SQL query statement, without any other explanation, comments, or formatting markers (such as ```sql)
"""
# Call the LLM to generate the SQL statement
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a SQL expert. Return only the SQL query statement, without any Markdown formatting or other explanation."},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

# Clean up the SQL statement, removing any Markdown markers
sql = response.choices[0].message.content.strip()
sql = sql.replace('```sql', '').replace('```', '').strip()
print(f"\nGenerated SQL query:\n{sql}")

# Execute the SQL and get the results
cursor.execute(sql)
results = cursor.fetchall()
print(f"Query results: {results}")
conn.close()

# Generate a natural language description
if results:
    # Get the column names
    column_names = [description[0] for description in cursor.description]
    # Convert the results into a list of dictionaries
    results_with_columns = [dict(zip(column_names, row)) for row in results]
    nl_prompt = f"""
The query results are as follows:
{results_with_columns}
Please convert this data into a natural language description that is easy to understand.
The original question was: {user_query}

Requirements:
1. Use clear, easy-to-understand language
2. Include all the queried data information
3. If there are numbers, spell them out in words
"""
    response_nl = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data analyst responsible for converting query results into an easy-to-understand natural language description."},
            {"role": "user", "content": nl_prompt}
        ],
        temperature=0.7
    )
    description = response_nl.choices[0].message.content.strip()
    print(f"Natural language description:\n{description}")
else:
    print("No relevant data found.")
# Close the database connection
conn.close()
