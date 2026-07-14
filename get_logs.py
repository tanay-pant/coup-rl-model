from huggingface_hub import HfApi
import requests

api = HfApi(token="***REMOVED***")
headers = {"Authorization": "Bearer ***REMOVED***"}
res = requests.get("https://huggingface.co/api/spaces/ptanay/coup-rl-backend/logs", headers=headers)
print(res.text[-2000:])
