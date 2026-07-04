from huggingface_hub import HfApi
api = HfApi(token="***REMOVED***")
api.restart_space("ptanay/coup-rl-backend")
print("Restarted Space!")
