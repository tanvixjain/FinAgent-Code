import os
import google.generativeai as genai

keys = {
    "ENV": os.getenv("GEMINI_API_KEY", ""),
    "CONFIG": "your_api_key_here"
}

with open("model_list_debug.txt", "w") as f:
    for name, key in keys.items():
        f.write(f"--- Testing {name} Key: {key[:10]}... ---\n")
        try:
            genai.configure(api_key=key)
            models = genai.list_models()
            for m in models:
                f.write(f"{m.name}\n")
        except Exception as e:
            f.write(f"Error: {e}\n")
        f.write("\n")
