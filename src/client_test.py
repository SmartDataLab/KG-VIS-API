import requests

res = requests.get("http://10.224.0.11:4002/neo/CASE")
print(res.text)