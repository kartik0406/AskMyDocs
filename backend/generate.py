from dotenv import load_dotenv
load_dotenv()

from google import genai
from config import *


client = genai.Client(api_key=GEMINI_API_KEY)

def generate_answer(query, docs):
    context = "\n\n".join(docs[:5])

    prompt = f"""
You are an intelligent assistant.

Answer the question based only on the context below.
- Keep answer concise (prefer bullet points)

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text