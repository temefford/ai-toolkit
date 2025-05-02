import os
from huggingface_hub import HfApi, login

# Configuration
CHECKPOINT_PATH = "output/benchmark/20250501-222935/benchmark.safetensors"
OPTIMIZER_PATH = "output/benchmark/20250501-222935/optimizer.pt"
REPO_ID = os.getenv("HF_REPO_ID")  # Set this env var to your repo, e.g. 'temefford/flux_schnell_baroque'
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

if not REPO_ID:
    raise RuntimeError("Set the HF_REPO_ID environment variable to your Hugging Face repo ID.")
if not HUGGINGFACE_TOKEN:
    raise RuntimeError("Set the HF_TOKEN environment variable to your Hugging Face token.")

login(token=HUGGINGFACE_TOKEN)
api = HfApi()

for file_path in [CHECKPOINT_PATH, OPTIMIZER_PATH]:
    print(f"Uploading {file_path} to {REPO_ID} ...")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=REPO_ID,
        repo_type="model",
        token=HUGGINGFACE_TOKEN,
        commit_message=f"Add {os.path.basename(file_path)}"
    )
    print(f"Uploaded {file_path}!")
