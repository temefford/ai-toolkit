import os
import json
from shutil import copyfile

baroque_dir = os.path.dirname(__file__)
gt_dir = os.path.join(baroque_dir, "gt")
os.makedirs(gt_dir, exist_ok=True)

with open(os.path.join(baroque_dir, "metadata.json")) as f:
    metadata = json.load(f)

for idx, entry in enumerate(metadata):
    img_hash = entry["hash"]
    found = False
    for ext in [".png", ".jpg", ".jpeg"]:
        src = os.path.join(baroque_dir, img_hash + ext)
        if os.path.exists(src):
            dst = os.path.join(gt_dir, f"{idx}.png")
            copyfile(src, dst)
            found = True
            break
    if not found:
        print(f"Image for hash {img_hash} not found!")

print(f"Copied ground-truth images to {gt_dir} as 0.png, 1.png, ...")
