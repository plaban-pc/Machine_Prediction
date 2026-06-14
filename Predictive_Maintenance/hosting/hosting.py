from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("PROJECT_TOKEN"))
api.upload_folder(
    folder_path="Predictive_Maintenance/deployment",
    repo_id="plaban95/Predictive-Maintenance",
    repo_type="space",
    path_in_repo="",
)
