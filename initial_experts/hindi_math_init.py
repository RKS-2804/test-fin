import os
import sys
import argparse
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

# Define the models to download and convert
models = [
    {
        "name": "OpenHathi-7B-Hi-v0.1-Base",
        "repo": "sarvamai/OpenHathi-7B-Hi-v0.1-Base",
        "output_dir": "openhathi-7b-hi-v0.1-base"
    },
    {
        "name": "WizardMath-7B-V1.1",
        "repo": "WizardLM/WizardMath-7B-V1.1",
        "output_dir": "wizardmath-7b-v1.1"
    },
    {
        "name": "Abel-7B-002",
        "repo": "GAIR/Abel-7B-002",
        "output_dir": "abel-7b-002"
    }
]

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
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

def download_and_convert_to_lora(model_info, use_auth=True, token=None):
    """Download a model and convert it to a LoRA adapter"""
    print(f"\n🚀 Processing {model_info['name']}...")
    
    # Create output directory
    os.makedirs(model_info['output_dir'], exist_ok=True)
    
    # Download model and tokenizer
    print(f"↓ Downloading {model_info['name']} from {model_info['repo']}...")
    try:
        # Set authentication parameters
        auth_kwargs = {}
        if use_auth and token:
            auth_kwargs = {"token": token}
        
        # Try to download the model
        model = AutoModelForCausalLM.from_pretrained(
            model_info['repo'],
            torch_dtype=torch.float16,
            device_map="auto",
            **auth_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_info['repo'], **auth_kwargs)
        
        # Convert to LoRA
        print(f"🔁 Converting {model_info['name']} to LoRA...")
        lora_model = get_peft_model(model, lora_config)
        
        # Save LoRA adapter and tokenizer
        print(f"💾 Saving {model_info['name']} LoRA adapter...")
        lora_model.save_pretrained(model_info['output_dir'])
        tokenizer.save_pretrained(model_info['output_dir'])
        
        # Clean up
        del model
        del lora_model
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

def main():
    parser = argparse.ArgumentParser(description="Download and prepare Hindi Math LLM models")
    parser.add_argument("--skip-auth", action="store_true", help="Skip Hugging Face authentication")
    parser.add_argument("--token", type=str, help="Hugging Face token (if not provided, will prompt)")
    args = parser.parse_args()
    
    print("📦 Starting Hindi Math LLM expert model prep...\n")
    
    # Check for PyTorch installation
    if not torch.__version__:
        print("❌ ERROR: PyTorch is not properly installed.")
        print("Please install PyTorch with CUDA support:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
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
    
    # Download and convert models
    success_count = 0
    for model_info in models:
        if download_and_convert_to_lora(model_info, use_auth, token if use_auth else None):
            success_count += 1
    
    # Summary
    print("\n=== Summary ===")
    print(f"Successfully processed {success_count} out of {len(models)} models.")
    
    if success_count == len(models):
        print("\n🎉 All models processed successfully!")
        print("You can now run the BWR_model search with these experts.")
    else:
        print("\n⚠️ Some models could not be processed automatically.")
        print("Please follow the alternative approach instructions for those models.")
        print("Once all models are ready, you can run the BWR_model search.")

if __name__ == "__main__":
    main()
