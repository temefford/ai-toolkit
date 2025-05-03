import torch
import numpy as np
from PIL import Image
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPProcessor, CLIPModel

# Utility to convert PIL image to normalized tensor
def pil_to_tensor(image: Image.Image, device: torch.device=None) -> torch.Tensor:
    image = image.convert('RGB')
    arr = np.array(image).transpose(2, 0, 1)  # C,H,W
    tensor = torch.from_numpy(arr).float() / 255.0
    if device:
        tensor = tensor.to(device)
    return tensor

# 1. Validation MSE between ground-truth and generated images
def compute_validation_mse(gt_images: list[Image.Image], gen_images: list[Image.Image]) -> float:
    gt_tensors = torch.stack([pil_to_tensor(img) for img in gt_images])
    gen_tensors = torch.stack([pil_to_tensor(img) for img in gen_images])
    mse = torch.nn.functional.mse_loss(gen_tensors, gt_tensors)
    return mse.item()

# 2. CLIP score between text prompts and generated images
def compute_clip_score(prompts: list[str], images: list[Image.Image], device: torch.device=None) -> float:
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    if device:
        model = model.to(device)
    inputs = processor(text=prompts, images=images, return_tensors='pt', padding=True)
    if device:
        inputs = {k:v.to(device) for k,v in inputs.items()}
    outputs = model(**inputs)
    img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    txt_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    sims = (img_embeds * txt_embeds).sum(dim=-1)
    return sims.mean().item()

# 3. Inception Score for generated images
def compute_inception_score(images: list[Image.Image], device: torch.device=None, splits: int=10) -> tuple[float, float]:
    # Convert and resize images to uint8 tensor [N,C,H,W]
    tensors = []
    for img in images:
        img_resized = img.convert('RGB').resize((299, 299), Image.BILINEAR)
        arr = np.array(img_resized).transpose(2, 0, 1)
        tensor = torch.from_numpy(arr)
        if device:
            tensor = tensor.to(device)
        tensors.append(tensor)
    batch = torch.stack(tensors)
    is_metric = InceptionScore(feature=2048).to(device) if device else InceptionScore(feature=2048)
    is_metric.update(batch)
    return is_metric.compute()

# 4. Fréchet Inception Distance between real and generated images
def compute_fid(gt_images: list[Image.Image], gen_images: list[Image.Image], device: torch.device=None) -> float:
    # Convert and resize images to uint8 tensor [N,C,H,W]
    gt_tensors = []
    for img in gt_images:
        img_resized = img.convert('RGB').resize((299, 299), Image.BILINEAR)
        arr = np.array(img_resized).transpose(2, 0, 1)
        tensor = torch.from_numpy(arr)
        if device:
            tensor = tensor.to(device)
        gt_tensors.append(tensor)
    gen_tensors = []
    for img in gen_images:
        img_resized = img.convert('RGB').resize((299, 299), Image.BILINEAR)
        arr = np.array(img_resized).transpose(2, 0, 1)
        tensor = torch.from_numpy(arr)
        if device:
            tensor = tensor.to(device)
        gen_tensors.append(tensor)
    gt_batch = torch.stack(gt_tensors)
    gen_batch = torch.stack(gen_tensors)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device) if device else FrechetInceptionDistance(feature=2048)
    fid_metric.update(gt_batch, real=True)
    fid_metric.update(gen_batch, real=False)
    return fid_metric.compute().item()
