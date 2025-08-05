import os
import sys
import logging
from pathlib import Path
import torch
from huggingface_hub import login, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All financial models from your list
FINANCIAL_MODELS = [
    {
        "repo_id": "Rakuten/RakutenAI-7B",
        "name": "RakutenAI-7B",
        "description": "Foundation model with Apache 2.0 license"
    },
    {
        "repo_id": "Rakuten/RakutenAI-7B-Instruct", 
        "name": "RakutenAI-7B-Instruct",
        "description": "Instruction-tuned variant optimized for instruction-following tasks"
    },
    {
        "repo_id": "Rakuten/RakutenAI-7B-Chat",
        "name": "RakutenAI-7B-Chat", 
        "description": "Chat-optimized version tailored for dialogue interactions"
    },
    {
        "repo_id": "elyza/Llama-3-ELYZA-JP-8B",
        "name": "Llama-3-ELYZA-JP-8B",
        "description": "Japanese fine-tuned model based on Meta's Llama-3 8B"
    },
    {
        "repo_id": "elyza/ELYZA-japanese-Llama-2-13B-instruct",
        "name": "ELYZA-japanese-Llama-2-13B-instruct", 
        "description": "Larger 13B instruct-tuned variant from the ELYZA family"
    },
    {
        "repo_id": "pfnet/nekomata-7b-pfn-qfin",
        "name": "nekomata-7B-pfn-qfin",
        "description": "Financial-domain adapted LLM built via continual pre-training"
    },
    {
        "repo_id": "pfnet/nekomata-14b-pfn-qfin", 
        "name": "nekomata-14B-pfn-qfin",
        "description": "Larger version for richer performance on finance tasks"
    },
    {
        "repo_id": "izumi-lab/bert-small-japanese-fin",
        "name": "bert-small-japanese-fin",
        "description": "Compact BERT model pre-trained with Japanese financial text"
    },
    {
        "repo_id": "izumi-lab/bert-base-japanese-fin-additional",
        "name": "bert-base-japanese-fin-additional", 
        "description": "Base-size BERT further tuned on finance domain corpora"
    },
    {
        "repo_id": "izumi-lab/llama-7B-japanese-lora-v0-5ep",
        "name": "llama-7B-japanese-lora-v0-5ep",
        "description": "LoRA-adapted Japanese LLaMA 7B"
    }
]

# LoRA configuration - This works for most models, but might need adjustment
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

def setup_hf_token():
    """Setup Hugging Face authentication"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN environment variable not found. You may encounter rate limits or access issues.")
        token = input("Enter your Hugging Face token (or press Enter to continue without authentication): ").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            hf_token = token
    
    if hf_token:
        try:
            login(token=hf_token)
            logger.info("Successfully authenticated with Hugging Face")
            return True, hf_token
        except Exception as e:
            logger.error(f"Failed to authenticate with Hugging Face: {e}")
            return False, None
    return False, None

def download_financial_model(model_info, base_dir="./financial_models"):
    """Download a single financial model"""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    repo_id = model_info["repo_id"]
    model_name = model_info["name"]
    local_dir = base_path / model_name
    
    # Skip if already exists
    if local_dir.exists() and any(local_dir.iterdir()):
        logger.info(f"✓ Model '{model_name}' already exists locally. Skipping download.")
        return True, local_dir
    
    try:
        logger.info(f"Downloading {model_name} ({repo_id})")
        logger.info(f"Description: {model_info['description']}")
        
        # Download using snapshot_download for efficiency
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        logger.info(f"✓ Successfully downloaded {model_name}")
        return True, local_dir
        
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {str(e)}")
        return False, None

def convert_to_lora(model_path, model_name, output_dir="./initial_experts", token=None):
    """Convert a model to a LoRA adapter"""
    # Create output directory path with model name
    lora_dir = f"{output_dir}/{model_name}-lora"
    os.makedirs(lora_dir, exist_ok=True)
    
    # Skip if already exists and has adapter_config.json
    if os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
        logger.info(f"✓ LoRA adapter for '{model_name}' already exists at {lora_dir}. Skipping conversion.")
        return True
    
    # Handle already-LoRA models
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        logger.info(f"Model {model_name} is already a LoRA adapter. Copying to {lora_dir}...")
        import shutil
        for item in os.listdir(model_path):
            s = os.path.join(model_path, item)
            d = os.path.join(lora_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        return True
    
    logger.info(f"Converting {model_name} to LoRA adapter...")
    
    try:
        # Set authentication parameters
        auth_kwargs = {}
        if token:
            auth_kwargs = {"token": token}
        
        # Try to load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            **auth_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, **auth_kwargs)
        
        # Convert to LoRA
        logger.info(f"Applying LoRA adapters to {model_name}...")
        lora_model = get_peft_model(model, lora_config)
        
        # Save LoRA adapter and tokenizer
        logger.info(f"Saving {model_name} LoRA adapter to {lora_dir}...")
        lora_model.save_pretrained(lora_dir)
        tokenizer.save_pretrained(lora_dir)
        
        # Clean up
        del model
        del lora_model
        torch.cuda.empty_cache()
        
        logger.info(f"✅ Successfully converted {model_name} to LoRA adapter")
        return True
    except Exception as e:
        logger.error(f"❌ Error converting {model_name} to LoRA: {str(e)}")
        return False

def main():
    """Download all financial models and convert them to LoRA adapters"""
    print("EDINET Financial Benchmark - Financial Models Downloader and LoRA Converter")
    print("=" * 80)
    
    # Check for PyTorch installation
    if not torch.__version__:
        print("❌ ERROR: PyTorch is not properly installed.")
        print("Please install PyTorch with CUDA support:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
    
    # Setup authentication
    authenticated, token = setup_hf_token()
    
    print(f"\nFound {len(FINANCIAL_MODELS)} financial models to process:")
    for i, model in enumerate(FINANCIAL_MODELS, 1):
        print(f"{i:2d}. {model['name']}")
        print(f"    {model['description']}")
    
    confirm = input(f"\nDownload and convert all {len(FINANCIAL_MODELS)} models? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Operation cancelled.")
        return
    
    # Make sure initial_experts directory exists
    os.makedirs("./initial_experts", exist_ok=True)
    
    # Process all models
    download_successful = 0
    conversion_successful = 0
    failed_models = []
    
    print(f"\nProcessing {len(FINANCIAL_MODELS)} models...")
    print("=" * 80)
    
    for i, model_info in enumerate(FINANCIAL_MODELS, 1):
        model_name = model_info["name"]
        print(f"\n[{i}/{len(FINANCIAL_MODELS)}] Processing {model_name}")
        
        # Step 1: Download the model
        download_success, model_path = download_financial_model(model_info)
        if download_success:
            download_successful += 1
            
            # Step 2: Convert to LoRA
            print(f"Converting {model_name} to LoRA adapter...")
            conversion_success = convert_to_lora(model_path, model_name, "./initial_experts", token)
            
            if conversion_success:
                conversion_successful += 1
                print(f"✅ {model_name} successfully downloaded and converted to LoRA")
            else:
                failed_models.append(f"{model_name} (conversion failed)")
                print(f"❌ Failed to convert {model_name} to LoRA")
        else:
            failed_models.append(f"{model_name} (download failed)")
            print(f"❌ Failed to download {model_name}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Processing Summary:")
    print(f"✓ Downloads successful: {download_successful}/{len(FINANCIAL_MODELS)}")
    print(f"✓ Conversions successful: {conversion_successful}/{len(FINANCIAL_MODELS)}")
    
    if failed_models:
        print(f"\nFailed models:")
        for model in failed_models:
            print(f"  - {model}")
    
    if conversion_successful > 0:
        print(f"\nLoRA adapters saved to: ./initial_experts/")
        print("You can now run the JAYA model search with these experts!")

if __name__ == "__main__":
    main()