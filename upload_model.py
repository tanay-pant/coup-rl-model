from huggingface_hub import HfApi

api = HfApi(token="***REMOVED***")
repo_id = "ptanay/coup-rl-backend"

print("Uploading scripts folder...")
api.upload_folder(
    folder_path="scripts",
    path_in_repo="scripts",
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
