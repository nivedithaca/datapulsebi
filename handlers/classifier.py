import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def classify_request(user_input: str, has_file: bool = False) -> str:
    if has_file:
        return "FILE_ANALYSIS"

    # Hard rule: "why" anywhere in the question always wins
    if "why" in user_input.lower():
        return "WHY_QUESTION"

    prompt = f"""
You are a request classifier for a data analytics assistant.

Classify the following request into exactly one of these categories:
- WHY_QUESTION: User is asking for an explanation, analysis, insight, or reason behind performance — even if they also mention rankings or comparisons. Examples: "why is X down", "what caused", "explain", "what's driving", "how did", "underperforming", "which region is struggling and why", "top states and why they perform well"
- SQL_PULL: User ONLY wants raw data pulled, filtered, or listed with no explanation needed (e.g. "show me all orders", "pull sales by region", "give me top 10 customers", "list orders above $5000")
- FILE_ANALYSIS: User uploaded a file and wants it analyzed
- UNKNOWN: Cannot determine the request type

Important: If the question asks both for a ranking AND an explanation/reason, classify as WHY_QUESTION.

Request: "{user_input}"

Reply with only the category name. Nothing else.
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    result = response.choices[0].message.content.strip().upper()
    valid = ["SQL_PULL", "WHY_QUESTION", "FILE_ANALYSIS", "UNKNOWN"]
    return result if result in valid else "UNKNOWN"
