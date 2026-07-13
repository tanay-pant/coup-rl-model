from huggingface_hub import HfApi

api = HfApi(token="***REMOVED***")
repo_id = "ptanay/coup-rl-backend"

print("Uploading main_policy to the old checkpoints_lstm_advanced folder...")
api.upload_folder(
    folder_path="checkpoints_lstm_advanced_v2/checkpoint_50000/policies/main_policy",
    path_in_repo="checkpoints_lstm_advanced/checkpoint_50000/policies/main_policy",
    repo_id=repo_id,
    repo_type="space"
)

print("Uploading backend/main.py...")
api.upload_file(
    path_or_fileobj="backend/main.py",
    path_in_repo="backend/main.py",
    repo_id=repo_id,
    repo_type="space"
)

print("Upload successful!")
