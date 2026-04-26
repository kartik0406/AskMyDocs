import google.generativeai as genai
from config import *

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")


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

    response = model.generate_content(prompt)
    return response.text