from huggingface_hub import HfApi

api = HfApi(token="***REMOVED***")
repo_id = "ptanay/coup-rl-backend"

api.upload_file(
    path_or_fileobj="backend/main.py",
    path_in_repo="backend/main.py",
    repo_id=repo_id,
    repo_type="space"
)

api.upload_file(
    path_or_fileobj="envs/coup/coup_env.py",
    path_in_repo="envs/coup/coup_env.py",
    repo_id=repo_id,
    repo_type="space"
)

api.upload_file(
    path_or_fileobj="envs/coup/game_logic.py",
    path_in_repo="envs/coup/game_logic.py",
    repo_id=repo_id,
    repo_type="space"
)

print("Upload successful!")
