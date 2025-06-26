import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # loads from .env file

my_api_key = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=my_api_key)

def generate_answer_gemini(query, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Based on the following document excerpts, answer the question:

Context:
{context}

Question: {query}
Answer:"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()