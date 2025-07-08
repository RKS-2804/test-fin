import os
from transformers import AutoModelForCausalLM
from huggingface_hub import login

# Log into Hugging Face (don't expose your token in public/shared code!)
login(token=os.getenv("HF_TOKEN"))

# Expert models you want to download
expert_models = [
    "code_alpaca", 
    "cot", 
    "flan_v2", 
    "gemini_alpaca", 
    "lima", 
    "oasst1", 
    "open_orca", 
    "science", 
    "sharegpt", 
    "wizardlm"
]

# Base namespace on Hugging Face
base_model_namespace = "bunsenfeng"

def download_experts():
    """
    Download initial expert models from HuggingFace.
    These are assumed to be LoRA adapters or fine-tuned models.
    """
    for model_name in expert_models:
        local_dir = model_name
        remote_repo = f"{base_model_namespace}/{model_name}"

        if os.path.exists(local_dir):
            print(f"[✓] Model '{model_name}' already exists locally. Skipping.")
            continue

        print(f"↓ Downloading model '{remote_repo}'...")

        try:
            model = AutoModelForCausalLM.from_pretrained(remote_repo)
            model.save_pretrained(local_dir)
            print(f"[✓] Model '{model_name}' downloaded and saved to '{local_dir}'.\n")
        except Exception as e:
            print(f"[!] Failed to download '{model_name}': {str(e)}\n")

if __name__ == "__main__":
    download_experts()
    print("All expert models attempted.")
