import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# Define the models to download
models = [
    {
        "name": "shisa-gamma-7b-v1",
        "repo": "AUGMXNT/shisa-gamma-7b-v1",
        "output_dir": "shisa-gamma-7b-v1"
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

def download_and_convert_to_lora(model_info):
    """Download a model and convert it to a LoRA adapter"""
    print(f"Processing {model_info['name']}...")
    
    # Create output directory
    os.makedirs(model_info['output_dir'], exist_ok=True)
    
    # Download model and tokenizer
    print(f"Downloading {model_info['name']}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_info['repo'],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_info['repo'])
    
    # Convert to LoRA
    print(f"Converting {model_info['name']} to LoRA...")
    model = prepare_model_for_kbit_training(model)
    lora_model = get_peft_model(model, lora_config)
    
    # Save LoRA adapter
    print(f"Saving {model_info['name']} LoRA adapter...")
    lora_model.save_pretrained(model_info['output_dir'])
    tokenizer.save_pretrained(model_info['output_dir'])
    
    # Free memory
    del model
    del lora_model
    torch.cuda.empty_cache()
    
    print(f"Finished processing {model_info['name']}")

def main():
    print("Starting Japanese Math LLM initial experts preparation...")
    
    for model_info in models:
        download_and_convert_to_lora(model_info)
    
    print("All models processed successfully!")
    print("You can now run the BWR_model search with these initial experts.")

if __name__ == "__main__":
    main()