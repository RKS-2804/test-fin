#!/usr/bin/env python3
"""
Test script to verify that adapter_config.json is properly copied during lora_merge
with directly_load_safetensors=1 option.

This script tests the fix for the issue where adapter_config.json wasn't being copied
when using the fast_merge option, causing the error:

ValueError: Unrecognized model in search/hindi_math_bwr_donut-ghkr6slj-fd966c464-7qdpn2025-07-10_08:41:02/candidate_0. 
Should have a `model_type` key in its config.json...
"""

import os
import shutil
import sys
from pathlib import Path

# Import the lora_merge function from merge.py
from merge import lora_merge, MergeMethod

def test_adapter_config_copy():
    """Test that adapter_config.json is copied during lora_merge with directly_load_safetensors=1"""
    
    # Setup test directories
    test_dir = Path("test_merge_output")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    source_dir1 = test_dir / "source1"
    source_dir2 = test_dir / "source2"
    output_dir = test_dir / "output"
    
    # Create test directories
    os.makedirs(source_dir1, exist_ok=True)
    os.makedirs(source_dir2, exist_ok=True)
    
    # Create dummy adapter_model.safetensors files
    with open(source_dir1 / "adapter_model.safetensors", "w") as f:
        f.write("dummy content for model 1")
    
    with open(source_dir2 / "adapter_model.safetensors", "w") as f:
        f.write("dummy content for model 2")
    
    # Create dummy adapter_config.json files with model_type
    with open(source_dir1 / "adapter_config.json", "w") as f:
        f.write('{"model_type": "llama", "other_config": "value1"}')
    
    with open(source_dir2 / "adapter_config.json", "w") as f:
        f.write('{"model_type": "llama", "other_config": "value2"}')
    
    # Call lora_merge with directly_load_safetensors=1
    print("Running lora_merge with directly_load_safetensors=1...")
    weights = [0.6, 0.4]
    lora_name_list = [str(source_dir1), str(source_dir2)]
    
    try:
        lora_merge(
            weights=weights,
            lora_name_list=lora_name_list,
            output_name=str(output_dir),
            gpu_id=0,
            directly_load_safetensors=1,
            merge_method=MergeMethod.WEIGHTED_AVERAGE
        )
        
        # Verify that both files were copied to the output directory
        adapter_model_exists = os.path.exists(output_dir / "adapter_model.safetensors")
        adapter_config_exists = os.path.exists(output_dir / "adapter_config.json")
        
        print(f"adapter_model.safetensors exists: {adapter_model_exists}")
        print(f"adapter_config.json exists: {adapter_config_exists}")
        
        if adapter_model_exists and adapter_config_exists:
            print("SUCCESS: Both adapter_model.safetensors and adapter_config.json were copied!")
            
            # Verify the content of adapter_config.json (should be from source_dir1)
            with open(source_dir1 / "adapter_config.json", "r") as f:
                source_content = f.read()
            
            with open(output_dir / "adapter_config.json", "r") as f:
                output_content = f.read()
            
            if source_content == output_content:
                print("SUCCESS: adapter_config.json content matches the first source model!")
            else:
                print("ERROR: adapter_config.json content does not match the first source model!")
                return False
            
            return True
        else:
            if not adapter_model_exists:
                print("ERROR: adapter_model.safetensors was not copied!")
            if not adapter_config_exists:
                print("ERROR: adapter_config.json was not copied!")
            return False
    
    except Exception as e:
        print(f"ERROR: An exception occurred during testing: {e}")
        return False
    finally:
        # Clean up test directories (comment this out if you want to inspect the files)
        # shutil.rmtree(test_dir)
        pass

if __name__ == "__main__":
    print("Testing adapter_config.json copy during lora_merge...")
    success = test_adapter_config_copy()
    
    if success:
        print("\nTEST PASSED: The fix for adapter_config.json copying is working correctly!")
        sys.exit(0)
    else:
        print("\nTEST FAILED: The fix for adapter_config.json copying is not working correctly!")
        sys.exit(1)