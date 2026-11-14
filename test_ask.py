import requests
import json

url = "http://127.0.0.1:5000/ask"
data = {"question": "How do I create a monthly budget?"}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
