from dotenv import load_dotenv
load_dotenv()

from google import genai
from config import *


client = genai.Client(api_key=GEMINI_API_KEY)

def generate_answer(query, docs):
    """Generate an answer with citation enforcement.
    Each source chunk is numbered [1], [2], etc.
    The LLM must cite which sources it used in its answer.
    """
    # Number each chunk as a citable source
    context_parts = []
    for i, doc in enumerate(docs[:8]):
        context_parts.append(f"[Source {i+1}]: {doc}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a precise document analysis assistant. Your job is to answer questions using ONLY the provided source passages from the user's uploaded document.

RULES:
1. Read ALL source passages carefully before answering.
2. Extract the EXACT information requested — quote directly from the document when possible.
3. You MUST cite your sources using [Source N] notation after each claim or piece of information.
4. If multiple sources support the same point, cite all of them: [Source 1][Source 3].
5. If the information exists anywhere in the sources, you MUST find and return it.
6. Only say "not found in the document" if you have thoroughly checked every source and the information is genuinely absent.
7. Keep answers concise and direct. Use bullet points for multiple items.
8. Do NOT make up or infer information that isn't explicitly in the sources.

SOURCES FROM DOCUMENT:
{context}

USER QUESTION:
{query}

ANSWER (with citations):"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text