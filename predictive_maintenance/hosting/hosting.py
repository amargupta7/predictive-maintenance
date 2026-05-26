
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

repo_id = "amarg7/predictive-maintenance-be"
repo_type = "space"

# Check if Space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating Space '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk="docker", private=False)

# Upload deployment folder
api.upload_folder(
    folder_path="predictive_maintenance/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo=""
)

print("Deployment pushed to Hugging Face Space.")
