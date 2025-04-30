# AI Toolkit by Ostris

AI Toolkit is an all in one training suite for diffusion models. I try to support all the latest models on consumer grade hardware. Image and video models. It can be run as a GUI or CLI. It is designed to be easy to use but still have every feature imaginable.

## Installation

Requirements:
- python >3.10
- Nvidia GPU with enough ram to do what you need
- python venv
- git


Linux:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
# install torch first
pip3 install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt
```

Windows:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```


## FLUX.1 Training

### Requirements
You currently need a GPU with **at least 24GB of VRAM** to train FLUX.1. If you are using it as your GPU to control 
your monitors, you probably need to set the flag `low_vram: true` in the config file under `model:`. This will quantize
the model on CPU and should allow it to train with monitors attached. Users have gotten it to work on Windows with WSL,
but there are some reports of a bug when running on windows natively. 
I have only tested on linux for now. This is still extremely experimental
and a lot of quantizing and tricks had to happen to get it to fit on 24GB at all. 

### FLUX.1-dev

FLUX.1-dev has a non-commercial license. Which means anything you train will inherit the
non-commercial license. It is also a gated model, so you need to accept the license on HF before using it.
Otherwise, this will fail. Here are the required steps to setup a license.

1. Sign into HF and accept the model access here [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
2. Make a file named `.env` in the root on this folder
3. [Get a READ key from huggingface](https://huggingface.co/settings/tokens/new?) and add it to the `.env` file like so `HF_TOKEN=your_key_here`

### FLUX.1-schnell

FLUX.1-schnell is Apache 2.0. Anything trained on it can be licensed however you want and it does not require a HF_TOKEN to train.
However, it does require a special adapter to train with it, [ostris/FLUX.1-schnell-training-adapter](https://huggingface.co/ostris/FLUX.1-schnell-training-adapter).
It is also highly experimental. For best overall quality, training on FLUX.1-dev is recommended.

To use it, You just need to add the assistant to the `model` section of your config file like so:

```yaml
      model:
        name_or_path: "black-forest-labs/FLUX.1-schnell"
        assistant_lora_path: "ostris/FLUX.1-schnell-training-adapter"
        is_flux: true
        quantize: true
```

You also need to adjust your sample steps since schnell does not require as many

```yaml
      sample:
        guidance_scale: 1  # schnell does not do guidance
        sample_steps: 4  # 1 - 4 works well
```

### Training
1. Copy the example config file located at `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` for schnell) to the `config` folder and rename it to `whatever_you_want.yml`
2. Edit the file following the comments in the file
3. Run the file like so `python run.py config/whatever_you_want.yml`

A folder with the name and the training folder from the config file will be created when you start. It will have all 
checkpoints and images in it. You can stop the training at any time using ctrl+c and when you resume, it will pick back up
from the last checkpoint.

IMPORTANT. If you press crtl+c while it is saving, it will likely corrupt that checkpoint. So wait until it is done saving

### Need help?

Please do not open a bug report unless it is a bug in the code. You are welcome to [Join my Discord](https://discord.gg/VXmU2f5WEU)
and ask for help there. However, please refrain from PMing me directly with general question or support. Ask in the discord
and I will answer when I can.


## Training in RunPod
Example RunPod template: **runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04**
> You need a minimum of 24GB VRAM, pick a GPU by your preference.

#### Example config ($0.5/hr):
- 1x A40 (48 GB VRAM)
- 19 vCPU 100 GB RAM

#### Custom overrides (you need some storage to clone FLUX.1, store datasets, store trained models and samples):
- ~120 GB Disk
- ~120 GB Pod Volume
- Start Jupyter Notebook

### 1. Setup
```
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub 
apt-get update && apt-get install -y libgl1
```
### 2. Upload your dataset
- Create a new folder in the root, name it `dataset` or whatever you like.
- Drag and drop your .jpg, .jpeg, or .png images and .txt files inside the newly created dataset folder.

### 3. Login into Hugging Face with an Access Token
- Get a READ token from [here](https://huggingface.co/settings/tokens) and request access to Flux.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Run ```huggingface-cli login``` and paste your token.

### 4. Training
- Copy an example config file located at ```config/examples``` to the config folder and rename it to ```whatever_you_want.yml```.
- Edit the config following the comments in the file.
- Change ```folder_path: "/path/to/images/folder"``` to your dataset path like ```folder_path: "/workspace/ai-toolkit/your-dataset"```.
- Run the file: ```python run.py config/whatever_you_want.yml``` or ```accelerate launch run.py config/schnell_config.yaml```.

### Screenshot from RunPod
<img width="1728" alt="RunPod Training Screenshot" src="https://github.com/user-attachments/assets/53a1b8ef-92fa-4481-81a7-bde45a14a7b5">