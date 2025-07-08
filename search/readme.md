# Search Results Directory

This directory will contain the results of BWR_model searches. Each search will create a subdirectory with a name based on the search parameters, typically starting with the dataset name.

## Directory Structure

When you run a search (e.g., `bash search_nlgraph.sh`), a directory will be created here with a structure like:

```
search/
└── nlgraph_[hostname]_[timestamp]/
    ├── args.txt                  # Command line arguments used for the search
    ├── log.txt                   # Log file with search progress
    ├── utility_scratchpad.json   # Utility values for all models during search
    ├── best/                     # Best model found during search
    ├── worst/                    # Worst model found during search
    └── candidate_*/              # Individual models in the population
```

## Key Files

- **utility_scratchpad.json**: Contains the utility values (performance metrics) for all models during the search process. This file is updated after each iteration.
- **log.txt**: Contains detailed logs of the search process, including initialization, iteration progress, and final results.
- **best/**: Contains the weights of the best model found during the search.

## Analyzing Results

To analyze the results of a search:

1. Check the `log.txt` file for overall metrics and search progress.
2. Examine `utility_scratchpad.json` to see how model performance evolved during the search.
3. The best model can be found in the `best/` directory.

## Using the Best Model

The best model found during the search is saved in the `best/` directory. You can load it using:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it")
model.load_adapter("search/[search_directory]/global_best")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")

# Use the model
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))