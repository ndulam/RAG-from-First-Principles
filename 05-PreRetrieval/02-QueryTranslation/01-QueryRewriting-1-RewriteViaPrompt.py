from openai import OpenAI
from os import getenv
# Initialize the OpenAI client, pointing it at the DeepSeek URL
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=getenv("DEEPSEEK_API_KEY")
)
def rewrite_query(question: str) -> str:
    """Use an LLM to rewrite the query"""
    prompt = """You are a game customer support agent, and you need to help the user rewrite their question.

Rules:
1. Remove irrelevant information (such as personal context or small talk)
2. Use precise game terminology
3. Preserve the core intent of the question
4. Convert vague questions into specific queries
Original question: {question}
Please give the rewritten query directly (without any prefix or explanation)."""
    # Use the DeepSeek model to rewrite the query
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt.format(question=question)}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()
# Start testing
query = "Um, so I just started playing this game, and it feels really hard. On the Putuo Mountain level, uh, I just can't get past it no matter what. What skill should I learn first? I'm new, please help!"
print(f"\nOriginal query: {query}")
print(f"Rewritten query: {rewrite_query(query)}")
