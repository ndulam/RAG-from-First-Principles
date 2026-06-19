from langchain_core.output_parsers import JsonOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
# Define output format
parser = JsonOutputParser()
prompt = PromptTemplate.from_template("Please return user information in JSON format: {query}")
# Call LLM and parse
llm = ChatDeepSeek(model="deepseek-chat")
output = llm(prompt.format(query="User ID 123"))
# Extract content from AIMessage
parsed_output = parser.parse(output.content)
print(parsed_output)