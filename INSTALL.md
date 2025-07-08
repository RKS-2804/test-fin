# Installation Guide for BWR_model

This guide will help you set up the environment for the BWR_model project, particularly for the Hindi Math LLM task.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git

## Step 1: Clone the Repository

If you haven't already cloned the repository:

```bash
git clone <repository-url>
cd BWR_model
```

## Step 2: Create and Activate a Virtual Environment

### On Windows:

```bash
python -m venv env
.\env\Scripts\activate
```

### On Linux/Mac:

```bash
python -m venv env
source env/bin/activate
```

## Step 3: Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

If you encounter issues with PyTorch, install it directly:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step 4: Hugging Face Authentication

You'll need a Hugging Face account and token to download the models:

1. Create an account at [Hugging Face](https://huggingface.co/join)
2. Generate a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Accept the model licenses by visiting each model page in your browser:
   - [sarvamai/OpenHathi-7B-Hi-v0.1-Base](https://huggingface.co/sarvamai/OpenHathi-7B-Hi-v0.1-Base)
   - [WizardLM/WizardMath-7B-V1.1](https://huggingface.co/WizardLM/WizardMath-7B-V1.1)
   - [GAIR/Abel-7B-002](https://huggingface.co/GAIR/Abel-7B-002)

## Step 5: Download and Prepare Models

Run the initialization script:

```bash
cd initial_experts
python hindi_math_init.py
```

When prompted, enter your Hugging Face token.

Alternatively, you can provide your token directly:

```bash
python hindi_math_init.py --token YOUR_TOKEN_HERE
```

If you encounter authentication issues, you can try:

```bash
# Skip authentication (may not work for gated models)
python hindi_math_init.py --skip-auth
```

## Step 6: Prepare the Dataset

Ensure you have the Hindi math dataset ready:

1. Check the template at `data/eval/mgsm_hindi.json`
2. Replace it with your actual dataset following the same format

## Step 7: Run the BWR Algorithm

Execute the search script:

```bash
cd ..  # Return to the main directory
bash search_hindi_math.sh
```

## Troubleshooting

### Authentication Issues

If you encounter authentication errors:

1. Verify your token is valid at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Ensure you've accepted the model licenses by visiting each model page
3. Try logging in manually:
   ```python
   from huggingface_hub import login
   login()  # This will prompt for your token
   ```

### PyTorch Installation Issues

If you see "None of PyTorch, TensorFlow >= 2.0, or Flax have been found":

1. Uninstall the current PyTorch installation:
   ```bash
   pip uninstall torch torchvision torchaudio
   ```
2. Install PyTorch with the correct CUDA version:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### CUDA Out of Memory

If you encounter CUDA out of memory errors:

1. Try using a smaller batch size in the search script
2. Use the `--step_length 0.5` parameter for more conservative updates
3. Run with fewer initial experts by modifying the script

## Additional Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [BWR_model Documentation](./README.md)