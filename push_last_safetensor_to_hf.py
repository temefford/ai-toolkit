import os
from pathlib import Path
from huggingface_hub import HfApi, login

# ==== CONFIGURATION ====
# Set these variables as needed
MODEL_FOLDER = "Baroque/_latent_cache/"  # Change to your model output folder if different
REPO_ID = os.getenv("HF_REPO_ID")  # e.g. 'temefford/flux_schnell_baroque'
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")  # Or set directly

# =======================

def find_last_safetensor(folder):
    safetensors = list(Path(folder).rglob("*.safetensors"))
    if not safetensors:
        raise FileNotFoundError(f"No .safetensors files found in {folder}")
    safetensors.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return safetensors[0]

def main():
    if not HUGGINGFACE_TOKEN:
        raise RuntimeError("Hugging Face token not set. Set HF_TOKEN env variable or edit script.")
    login(token=HUGGINGFACE_TOKEN)
    last_model = find_last_safetensor(MODEL_FOLDER)
    print(f"Pushing {last_model} to HuggingFace repo {REPO_ID}...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(last_model),
        path_in_repo=last_model.name,
        repo_id=REPO_ID,
        repo_type="model",
        token=HUGGINGFACE_TOKEN,
        commit_message=f"Add latest fine-tuned model: {last_model.name}"
    )
    print("Upload complete!")

if __name__ == "__main__":
    main()
