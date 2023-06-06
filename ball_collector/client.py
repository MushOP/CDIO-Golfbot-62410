import requests
import json

response = requests.get('http://127.0.0.1:5000')
data = json.loads(response.text)

print(data)