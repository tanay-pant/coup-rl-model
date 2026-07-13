from huggingface_hub import HfApi

api = HfApi(token="***REMOVED***")
repo_id = "ptanay/coup-rl-backend"

print("Uploading repository to Hugging Face Space...")
api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=[
        "venv*",
        ".git*",
        "frontend/node_modules*",
        "plots*",
        "__pycache__*",
        "*.png",
        "get_logs.py",
        "upload_model.py",
        "upload_to_hf.py"
    ]
)
print("Upload successful!")
