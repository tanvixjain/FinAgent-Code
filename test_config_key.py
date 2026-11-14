import os
import google.generativeai as genai

# Test key from config_example.py
TEST_KEY = "your_api_key_here"
print(f"Testing Key: {TEST_KEY}")

genai.configure(api_key=TEST_KEY)

print("Listing all models for this key:")
try:
    for m in genai.list_models():
        print(f" - {m.name}")
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("ping")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
