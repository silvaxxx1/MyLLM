# MyLLM Data Directory

This directory contains scripts and utilities for handling and processing data used in training a GPT-like language model.

## Directory Structure

```
MyLLM/
└── data/
    ├── dataloader.py           # DataLoader for the GPTDataset
    ├── preprocess.py            # Functions for preprocessing raw data
    ├── data_test.py             # Tests for data processing scripts
    └── tests/                   # Unit tests for dataset and preprocessing functions
        ├── test_gpt_dataset.py  # Tests for GPTDataset class
        └── test_preprocess.py    # Tests for preprocessing functions
```

## Usage

- **Dataset and DataLoader**: Use the `GPTDataset` and `GPTDataLoader` classes to prepare data for model training.
- **Preprocessing**: The `preprocess.py` script contains functions for cleaning and preparing raw data.

## Testing

To run tests, use:

```bash
python -m unittest discover -s tests
```
