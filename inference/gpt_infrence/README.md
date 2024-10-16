---
title: gpt_infrence
app_file: gpt_app.py
sdk: gradio
sdk_version: 4.44.1
---
```markdown
# GPT Model Inference Directory

This directory contains scripts for loading and generating text with GPT model

## Scripts

### `load_model.py`
Utility for loading GPT models from checkpoint files. It provides the `load_model` function to load models and set them to evaluation mode.

### `gpt_infrence.py`
Command-line interface for generating text from prompts using a specified GPT model. 

**Usage:**
```bash
python gpt_infrence.py --prompt "Your prompt here" --max_length 100 --temperature 0.7 --top_k 50 --model_name "gpt2-small (124M)"
```

### `gpt_app.py`
Gradio web application for interactive text generation using GPT models.

**Run the App:**
```bash
python gpt_app.py
```
Access the app via the URL displayed in the terminal.

## Requirements
- `torch`
- `tiktoken`
- `gradio`

Install dependencies:
```bash
pip install torch tiktoken gradio
```
