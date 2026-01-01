"""
Test script to verify Google Gemini API key is set correctly.

Usage:
    python test_gemini_key.py

Expected output: "OK"
"""

import os
from google import genai

# Check if API key is set
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ ERROR: GEMINI_API_KEY environment variable is not set!")
    print("\nTo set it:")
    print("  Windows PowerShell: $env:GEMINI_API_KEY = 'YOUR_API_KEY'")
    print("  Linux/Mac: export GEMINI_API_KEY='YOUR_API_KEY'")
    print("\nGet your API key from: https://aistudio.google.com/apikey")
    exit(1)

print(f"✅ API Key found: {api_key[:10]}...")

try:
    client = genai.Client()   # reads GEMINI_API_KEY automatically
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with exactly: OK"
    )
    result = response.text.strip()
    if result == "OK":
        print("✅ API Key is valid! Quiz generation should work.")
    else:
        print(f"⚠️  Unexpected response: {result}")
except Exception as e:
    print(f"❌ ERROR: API key test failed: {e}")
    print("\nPossible issues:")
    print("  1. API key is invalid or expired")
    print("  2. No internet connection")
    print("  3. API quota exceeded")
    exit(1)
