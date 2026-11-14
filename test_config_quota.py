import os
import google.generativeai as genai

# Test key from config_example.py
TEST_KEY = "your_api_key_here"
print(f"Testing Key: {TEST_KEY}")

genai.configure(api_key=TEST_KEY)

try:
    # Use one that we know exists from the debug list
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content("ping")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
