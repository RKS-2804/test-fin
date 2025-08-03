import os
import logging
from pathlib import Path
from huggingface_hub import login, snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_hf_token():
    """Setup Hugging Face authentication"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN environment variable not found. You may encounter rate limits or access issues.")
        return False
    
    try:
        login(token=hf_token)
        logger.info("Successfully authenticated with Hugging Face")
        return True
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {e}")
        return False

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

def download_financial_model(model_info, base_dir="./financial_models"):
    """Download a single financial model"""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    repo_id = model_info["repo_id"]
    model_name = model_info["name"]
    local_dir = base_path / model_name
    
    # Skip if already exists
    if local_dir.exists() and any(local_dir.iterdir()):
        logger.info(f"✓ Model '{model_name}' already exists locally. Skipping.")
        return True
    
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
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {str(e)}")
        return False

def main():
    """Download all financial models for EDINET benchmark"""
    print("EDINET Financial Benchmark - Financial Models Downloader")
    print("=" * 60)
    
    setup_hf_token()
    
    print(f"\nFound {len(FINANCIAL_MODELS)} financial models to download:")
    for i, model in enumerate(FINANCIAL_MODELS, 1):
        print(f"{i:2d}. {model['name']}")
        print(f"    {model['description']}")
    
    confirm = input(f"\nDownload all {len(FINANCIAL_MODELS)} models? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Download cancelled.")
        return
    
    # Download all models
    successful = 0
    failed = 0
    failed_models = []
    
    print(f"\nStarting download of {len(FINANCIAL_MODELS)} models...")
    print("=" * 60)
    
    for i, model_info in enumerate(FINANCIAL_MODELS, 1):
        print(f"\n[{i}/{len(FINANCIAL_MODELS)}] Processing {model_info['name']}")
        if download_financial_model(model_info):
            successful += 1
        else:
            failed += 1
            failed_models.append(model_info['name'])
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    
    if failed_models:
        print(f"\nFailed models:")
        for model in failed_models:
            print(f"  - {model}")
    
    if successful > 0:
        print(f"\nModels saved to: ./financial_models/")
        print("You can now use these models for your EDINET financial benchmark!")

if __name__ == "__main__":
    main()