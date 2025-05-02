import os
import modal
from pathlib import Path

# Initialize the Modal app
app = modal.App("stable-diffusion-finetuning")

# Volume to persist model and dataset files across runs
volume = modal.Volume.from_name("stable-diffusion-data-vol", create_if_missing=True)

# Custom image setup with all required libraries and tools
gpu_image = modal.Image.debian_slim()\
    .apt_install("git")\
    .pip_install(
        "diffusers==0.25.0",
        "transformers",
        "accelerate",
        "torch",
        "torchvision",
        "pillow",
        "tqdm",
        "datasets",
        "peft",
        "bitsandbytes",
        "safetensors",
        "huggingface_hub",
    )\
    .run_commands("pip install git+https://github.com/huggingface/diffusers")

# HuggingFace token handled through Modal Secrets
hf_secret = modal.Secret.from_name("hf-token")

@app.function(
    image=gpu_image,
    volumes={"/data": volume},
    timeout=3600,
    secrets=[hf_secret]
)
def download_dataset():
    # Downloads ad-style image dataset from HuggingFace and saves locally
    from datasets import load_dataset
    from tqdm import tqdm
    from huggingface_hub import login

    dataset_dir = "/data/ad_dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    # Pull HuggingFace token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HuggingFace token (HF_TOKEN) not found in environment.")

    login(hf_token)
    print("Logged in to HuggingFace")

    print("Downloading dataset...")
    dataset = load_dataset("PeterBrendan/AdImageNet")

    print("Saving images locally...")
    for i, sample in enumerate(tqdm(dataset['train'])):
        sample['image'].save(f"{dataset_dir}/{i}.png")

    print(f"Dataset saved at: {dataset_dir}")
    return dataset_dir

@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={"/data": volume},
    timeout=7200,
    secrets=[hf_secret]
)
def train_lora_model():
    # Fine-tunes Stable Diffusion with LoRA on the dataset
    import subprocess
    from huggingface_hub import login

    dataset_dir = "/data/ad_dataset"
    output_dir = "/data/ad-style-lora"

    # Auth again
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HuggingFace token (HF_TOKEN) not found in environment.")

    login(hf_token)
    print("Logged in to HuggingFace")

    # Download dataset if it’s not there yet
    if not os.path.exists(dataset_dir):
        print("Dataset not found, downloading it...")
        download_dataset.remote()
    else:
        print(f"Found dataset at: {dataset_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Clone training scripts if not already available
    if not os.path.exists("/data/diffusers"):
        print("Cloning diffusers repo...")
        subprocess.run(["git", "clone", "https://github.com/huggingface/diffusers", "/data/diffusers"], check=True)

    # Launch the LoRA training script
    print("Starting training...")
    cmd = [
        "accelerate", "launch", "/data/diffusers/examples/dreambooth/train_dreambooth_lora.py",
        "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
        f"--instance_data_dir={dataset_dir}",
        f"--output_dir={output_dir}",
        "--instance_prompt=a stylish advertising banner",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--max_train_steps=1000",
        "--checkpointing_steps=500",
        "--use_8bit_adam",
        "--mixed_precision=fp16",
        "--report_to=tensorboard"
    ]

    subprocess.run(cmd, cwd="/data/diffusers/examples/dreambooth", check=True)
    print(f"Model fine-tuned and saved at: {output_dir}")
    return output_dir

@app.function(
    image=gpu_image,
    gpu="A10G",
    volumes={"/data": volume},
    timeout=3600
)
def generate_with_finetuned(prompt: str = "A stylish shoe advertisement with clean background"):
    # Runs inference with the fine-tuned model using the provided prompt
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image
    import io, base64

    lora_model_path = "/data/ad-style-lora"
    if not os.path.exists(lora_model_path):
        return {"error": "Model not found. Please train it first."}

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs(lora_model_path)
    pipe.to("cuda")

    print(f"Generating with prompt: {prompt}")
    image = pipe(prompt, num_inference_steps=50).images[0]

    os.makedirs("/data/outputs", exist_ok=True)
    image_path = "/data/outputs/finetuned_output.png"
    image.save(image_path)

    # Convert image to base64 for return
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"image_path": image_path, "image_data": img_str}

@app.local_entrypoint()
def main(task: str = "all", prompt: str = None):
    # CLI-like entrypoint to control which parts of the pipeline to run
    try:
        if task in ["all", "download"]:
            print(">>> Downloading dataset")
            download_dataset.remote()

        if task in ["all", "train"]:
            print(">>> Starting model fine-tuning")
            train_lora_model.remote()

        if task in ["all", "inference"] or (task == "all" and prompt):
            print(">>> Generating image with fine-tuned model")
            result = generate_with_finetuned.remote(prompt=prompt or "A stylish shoe advertisement with clean background")
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Image generated at: {result['image_path']}")

        print("All done.")
    except Exception as e:
        print(f"Pipeline error: {str(e)}")
