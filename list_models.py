import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
print(f"API Key: {GEMINI_API_KEY}")

genai.configure(api_key=GEMINI_API_KEY)

print("Listing all models:")
try:
    models = genai.list_models()
    for m in models:
        print(f" - {m.name} ({m.supported_generation_methods})")
except Exception as e:
    print(f"Error listing models: {e}")
