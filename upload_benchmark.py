#!/usr/bin/env python3
"""
Script to upload a large folder to Hugging Face model repo.
"""
import os
from huggingface_hub import HfApi
from pathlib import Path

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: Please set the HF_TOKEN environment variable.")
        return

    api = HfApi(token=token)
    repo_id = "TheoMefff/flux_schnell_baroque_rackspace_pvc"
    folder_path = "output/benchmark"

    print(f"Uploading {folder_path} to {repo_id}...")
    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        repo_type="model"
    )
    print("Upload complete!")

    # Upload latest markdown results as results.md
    md_dir = Path("results/benchmarks")
    md_files = list(md_dir.glob("*.md"))
    if md_files:
        latest_md = max(md_files, key=lambda p: p.stat().st_mtime)
        print(f"Uploading markdown {latest_md} to {repo_id} as results.md")
        api.upload_file(
            path_or_fileobj=str(latest_md),
            path_in_repo="results.md",
            repo_id=repo_id,
            repo_type="model"
        )
        print("Markdown upload complete!")
    else:
        print("No markdown files found to upload.")

if __name__ == "__main__":
    main()
