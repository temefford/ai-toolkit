#!/usr/bin/env python3
"""
Script to upload a large folder to Hugging Face model repo.
"""
import os
from huggingface_hub import HfApi

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: Please set the HF_TOKEN environment variable.")
        return

    api = HfApi(token=token)
    repo_id = "TheoMefff/flux_schnell_baroque_rackspace"
    folder_path = "output/benchmark"

    print(f"Uploading {folder_path} to {repo_id}...")
    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        repo_type="model"
    )
    print("Upload complete!")

if __name__ == "__main__":
    main()
