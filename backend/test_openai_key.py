import os
from openai import OpenAI

key = os.getenv("OPENAI_API_KEY")
print("Key loaded:", "YES" if key else "NO")

client = OpenAI(api_key=key)

response = client.responses.create(
    model="gpt-5-mini",
    input="Reply with exactly: OK"
)

print(response.output_text)
