import os
import sys
import argparse
import torch
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig

# Embedding expert models you want to download (JP + EN)
embedding_models = [
    # Japanese embedding models
    {
        "name": "ruri-large",
        "repo": "cl-nagoya/ruri-large",
        "output_dir": "cl-nagoya_ruri-large"
    },
    {
        "name": "ruri-base", 
        "repo": "cl-nagoya/ruri-base",
        "output_dir": "cl-nagoya_ruri-base"
    },
    {
        "name": "multilingual-e5-large-instruct",
        "repo": "intfloat/multilingual-e5-large-instruct",
        "output_dir": "intfloat_multilingual-e5-large-instruct"
    },
    {
        "name": "GLuCoSE-base-ja-v2",
        "repo": "pkshatech/GLuCoSE-base-ja-v2", 
        "output_dir": "pkshatech_GLuCoSE-base-ja-v2"
    },
    {
        "name": "bge-m3",
        "repo": "BAAI/bge-m3",
        "output_dir": "BAAI_bge-m3"
    },
    
    # English & Multilingual embedding models (MTEB leaders)
    {
        "name": "NV-Embed-v2",
        "repo": "nvidia/NV-Embed-v2",
        "output_dir": "nvidia_NV-Embed-v2"
    },
    {
        "name": "bge-en-icl",
        "repo": "BAAI/bge-en-icl",
        "output_dir": "BAAI_bge-en-icl"
    },
    {
        "name": "SFR-Embedding-2_R", 
        "repo": "Salesforce/SFR-Embedding-2_R",
        "output_dir": "Salesforce_SFR-Embedding-2_R"
    },
    {
        "name": "multilingual-e5-large",
        "repo": "intfloat/multilingual-e5-large",
        "output_dir": "intfloat_multilingual-e5-large"
    },
    {
        "name": "gecko-multilingual",
        "repo": "google/gecko-multilingual", 
        "output_dir": "google_gecko-multilingual"
    }
]

# LoRA configuration for embedding models
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense", "q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)

def authenticate_huggingface():
    """Interactive authentication with Hugging Face Hub"""
    print("\n=== Hugging Face Authentication ===")
    print("You need to authenticate with Hugging Face to download the models.")
    print("Please visit: https://huggingface.co/settings/tokens")
    print("Create a new token with 'read' access and enter it below.")
    
    token = input("Enter your Hugging Face token: ").strip()
    
    try:
        login(token=token)
        print("✅ Successfully authenticated with Hugging Face!")
        return True, token
    except Exception as e:
        print(f"❌ Authentication failed: {str(e)}")
        print("\nPlease ensure you have:")
        print("1. A valid Hugging Face account")
        print("2. Generated a valid access token at https://huggingface.co/settings/tokens")
        print("3. Accepted the model licenses if required (visit the model pages in a browser)")
        return False, None

def download_and_convert_to_lora(model_info, use_auth=True, token=None, convert_to_lora=False):
    """Download an embedding model and optionally convert it to a LoRA adapter"""
    print(f"\n🚀 Processing {model_info['name']}...")
    
    # Create output directory
    os.makedirs(model_info['output_dir'], exist_ok=True)
    
    # Check if already exists
    if os.path.exists(model_info['output_dir']) and len(os.listdir(model_info['output_dir'])) > 0:
        print(f"[✓] Model '{model_info['name']}' already exists locally at '{model_info['output_dir']}'. Skipping.")
        return True
    
    # Download model and tokenizer
    print(f"↓ Downloading {model_info['name']} from {model_info['repo']}...")
    try:
        # Set authentication parameters
        auth_kwargs = {}
        if use_auth and token:
            auth_kwargs = {"token": token}
        
        # Try to download the model
        print(f"  ↓ Downloading model...")
        model = AutoModel.from_pretrained(
            model_info['repo'],
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            **auth_kwargs
        )
        
        print(f"  ↓ Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_info['repo'], 
            trust_remote_code=True,
            **auth_kwargs
        )
        
        if convert_to_lora:
            # Convert to LoRA
            print(f"🔁 Converting {model_info['name']} to LoRA...")
            try:
                lora_model = get_peft_model(model, lora_config)
                
                # Save LoRA adapter and tokenizer
                print(f"💾 Saving {model_info['name']} LoRA adapter...")
                lora_model.save_pretrained(model_info['output_dir'])
                tokenizer.save_pretrained(model_info['output_dir'])
                
                # Clean up
                del lora_model
            except Exception as lora_e:
                print(f"⚠️ LoRA conversion failed for {model_info['name']}: {str(lora_e)}")
                print(f"💾 Saving original model instead...")
                model.save_pretrained(model_info['output_dir'])
                tokenizer.save_pretrained(model_info['output_dir'])
        else:
            # Save original model
            print(f"💾 Saving {model_info['name']} original model...")
            model.save_pretrained(model_info['output_dir'])
            tokenizer.save_pretrained(model_info['output_dir'])
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✅ Finished {model_info['name']}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {model_info['name']}: {str(e)}")
        print("\nAlternative approach:")
        print(f"1. Visit {model_info['repo']} in your browser")
        print("2. Download the model files manually")
        print(f"3. Place them in the {model_info['output_dir']} directory")
        return False

def list_downloaded_models():
    """List all downloaded embedding models in the current directory."""
    print("\n" + "="*60)
    print("DOWNLOADED EMBEDDING MODELS (JP + EN):")
    print("="*60)
    
    for model_info in embedding_models:
        local_dir = model_info['output_dir']
        if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
            try:
                size = sum(os.path.getsize(os.path.join(local_dir, f)) 
                          for f in os.listdir(local_dir) 
                          if os.path.isfile(os.path.join(local_dir, f)))
                size_mb = size / (1024 * 1024)
                print(f"[✓] {model_info['repo']} → {local_dir}/ ({size_mb:.1f} MB)")
            except:
                print(f"[✓] {model_info['repo']} → {local_dir}/ (size calculation failed)")
        else:
            print(f"[✗] {model_info['repo']} → Not downloaded")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare Japanese & English embedding models")
    parser.add_argument("--skip-auth", action="store_true", help="Skip Hugging Face authentication")
    parser.add_argument("--token", type=str, help="Hugging Face token (if not provided, will prompt)")
    parser.add_argument("--convert-lora", action="store_true", help="Convert models to LoRA adapters")
    parser.add_argument("--models", nargs='+', help="Specific models to download (by name)")
    args = parser.parse_args()
    
    print("🇯🇵🇺🇸 Japanese & English Embedding Models Downloader")
    print("="*70)
    
    # Check for PyTorch installation
    try:
        torch_version = torch.__version__
        print(f"✅ PyTorch {torch_version} detected")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA not available, using CPU")
    except:
        print("❌ ERROR: PyTorch is not properly installed.")
        print("Please install PyTorch: pip install torch torchvision torchaudio")
        sys.exit(1)
    
    # Authenticate with Hugging Face
    use_auth = not args.skip_auth
    token = args.token
    
    if use_auth:
        if not token:
            auth_success, token = authenticate_huggingface()
            if not auth_success:
                print("\nWould you like to:")
                print("1. Try authentication again")
                print("2. Continue without authentication (may fail for gated models)")
                print("3. Exit")
                choice = input("Enter your choice (1/2/3): ").strip()
                
                if choice == "1":
                    auth_success, token = authenticate_huggingface()
                    if not auth_success:
                        print("❌ Authentication failed again. Exiting.")
                        sys.exit(1)
                elif choice == "2":
                    use_auth = False
                    print("Continuing without authentication...")
                else:
                    print("Exiting.")
                    sys.exit(1)
        else:
            try:
                login(token=token)
                print("✅ Successfully authenticated with Hugging Face using provided token!")
            except Exception as e:
                print(f"❌ Authentication failed with provided token: {str(e)}")
                auth_success, token = authenticate_huggingface()
                if not auth_success:
                    print("❌ Authentication failed again. Exiting.")
                    sys.exit(1)
    
    # Filter models if specific ones requested
    models_to_download = embedding_models
    if args.models:
        models_to_download = [m for m in embedding_models if m['name'] in args.models]
        print(f"📋 Downloading specific models: {', '.join(args.models)}")
    
    # Download and convert models
    success_count = 0
    for model_info in models_to_download:
        if download_and_convert_to_lora(model_info, use_auth, token if use_auth else None, args.convert_lora):
            success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("=== SUMMARY ===")
    print("="*60)
    print(f"Successfully processed {success_count} out of {len(models_to_download)} models.")
    
    if success_count == len(models_to_download):
        print("\n🎉 All models processed successfully!")
        if args.convert_lora:
            print("LoRA adapters are ready for merging.")
        else:
            print("Original models are ready for use.")
    else:
        print("\n⚠️ Some models could not be processed automatically.")
        print("Please follow the alternative approach instructions for those models.")
    
    list_downloaded_models()
    
    print(f"\n💡 Usage tip: These models are optimized for:")
    print("   🇯🇵 JAPANESE:")
    print("   • ruri-large/base: JSTS, MIRACL, general Japanese tasks")
    print("   • multilingual-e5-large-instruct: Multilingual + Japanese")
    print("   • GLuCoSE-base-ja-v2: Japanese correlation tasks") 
    print("   • bge-m3: Multimodal embeddings with Japanese support")
    print("   🇺🇸 ENGLISH/MULTILINGUAL:")
    print("   • NV-Embed-v2: NVIDIA's MTEB leader")
    print("   • bge-en-icl: BGE family, retrieval + classification")
    print("   • SFR-Embedding-2_R: Salesforce, reranking + retrieval")
    print("   • multilingual-e5-large: Universal multilingual")
    print("   • gecko-multilingual: Compact, efficient, outperforms larger models")

if __name__ == "__main__":
    main()