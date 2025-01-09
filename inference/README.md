# GPT Model Inference Directory

Welcome to the **GPT Model Inference Directory**! This repository provides scripts for loading and generating text with GPT models, allowing users to easily interact with various GPT architectures.

<p align="center">
    <img src="capturex.png" alt="My Image" />
</p>

## Overview

This directory contains essential scripts that facilitate the loading of GPT models and the generation of text based on user prompts. Whether you're running inference from the command line or through a web interface, we've got you covered.

## Scripts

### `load_model.py`
A utility script for loading GPT models from checkpoint files. This script includes the `load_model` function, which loads the specified model and sets it to evaluation mode for inference.

### `gpt_inference.py`
This command-line interface (CLI) allows you to generate text from prompts using a specified GPT model. 

**Usage:**
```bash
python gpt_inference.py --prompt "Your prompt here" --max_length 100 --temperature 0.7 --top_k 50 --model_name "gpt2-small (124M)"
```
You can customize the prompt, maximum length, temperature, top-k sampling, and model name to suit your needs.

### `gpt_app.py`
A Gradio web application that provides an interactive interface for text generation using GPT models.

**Run the App:**
```bash
python gpt_app.py
```
Once the app is running, access it via the URL displayed in your terminal to start generating text interactively.

## Requirements
To run these scripts, you will need the following Python packages:

- `torch`
- `tiktoken`
- `gradio`

### Installation
You can install the required dependencies using pip:

```bash
pip install torch tiktoken gradio
```

## Conclusion

With these scripts, you can easily load GPT models and generate text in both command-line and web environments. Feel free to explore and modify the scripts to enhance your text generation projects!
