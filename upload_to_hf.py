from huggingface_hub import HfApi
import os

api = HfApi(token="***REMOVED***")
repo_id = "ptanay/coup-rl-backend"

print("Uploading repository core files to Hugging Face Space...")
api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=[
        "checkpoints*",
        "venv*",
        ".git*",
        "frontend/node_modules*",
        "__pycache__*",
        "*.png",
        "get_logs.py",
        "upload_model.py",
        "upload_to_hf.py"
    ]
)

print("Uploading only checkpoint_50000 to Hugging Face Space...")
ckpt_path = "checkpoints_lstm_advanced_v2/checkpoint_50000"
if os.path.exists(ckpt_path):
    api.upload_folder(
        folder_path=ckpt_path,
        path_in_repo=ckpt_path,
        repo_id=repo_id,
        repo_type="space"
    )

print("Upload successful!")
