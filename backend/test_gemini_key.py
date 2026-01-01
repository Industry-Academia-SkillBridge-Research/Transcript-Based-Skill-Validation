from google import genai

client = genai.Client()   # reads GEMINI_API_KEY automatically
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with exactly: OK"
)
print(response.text)
