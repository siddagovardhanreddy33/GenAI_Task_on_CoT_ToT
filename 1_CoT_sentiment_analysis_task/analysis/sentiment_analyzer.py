from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
apikey = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=apikey)


def analyze_review(prompt):
    response = client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
    return response.text