import os
import logging
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from trl.core import LengthSampler
import torch
import json

# ---------------------------
# Logging Configuration
# ---------------------------
def setup_logging(log_dir="logs", log_file="data_processing.log"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")

# ---------------------------
# Dataset Utilities
# ---------------------------
def load_and_filter_dataset(dataset_name: str, min_review_length: int):
    """
    Load a dataset and filter reviews by minimum length.
    
    Args:
        dataset_name (str): Name of the dataset to load.
        min_review_length (int): Minimum character length for reviews.
    
    Returns:
        pd.DataFrame: Filtered dataset as a DataFrame.
    """
    logging.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset['train'])
    df = df.rename(columns={'text': 'review'})
    df = df[df['review'].apply(lambda x: len(x) > min_review_length)]
    logging.info(f"Dataset loaded and filtered. {len(df)} samples remaining.")
    return df

def tokenize_dataset(df, tokenizer, input_min_text_length, input_max_text_length):
    """
    Tokenize a dataset using the specified tokenizer and text length sampler.
    
    Args:
        df (pd.DataFrame): DataFrame containing the dataset to tokenize.
        tokenizer (AutoTokenizer): Tokenizer instance.
        input_min_text_length (int): Minimum tokenized text length.
        input_max_text_length (int): Maximum tokenized text length.
    
    Returns:
        pd.DataFrame: Tokenized DataFrame.
    """
    logging.info("Initializing tokenization...")
    length_sampler = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize_row(row):
        input_size = length_sampler()
        input_ids = tokenizer.encode(row["review"], truncation=True, max_length=input_size, padding=False)
        query = tokenizer.decode(input_ids)
        return pd.Series({"input_ids": input_ids, "query": query})

    df[['input_ids', 'query']] = df.apply(lambda row: tokenize_row(row), axis=1)
    df['input_ids'] = df['input_ids'].apply(torch.tensor)  # Convert to tensor
    logging.info("Tokenization complete.")
    return df

class CustomDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for tokenized data.
    """
    def __init__(self, dataframe):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing tokenized data.
        """
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.
        
        Returns:
            dict: A dictionary containing input_ids and query.
        """
        row = self.dataframe.iloc[idx]
        return {
            "input_ids": row["input_ids"],
            "query": row["query"]
        }

def save_dataset_to_file(dataset, save_dir="processed_data", save_file="tokenized_data.json"):
    """
    Save the processed dataset to a file for later use.

    Args:
        dataset (CustomDataset): The PyTorch Dataset object.
        save_dir (str): Directory where the dataset will be saved.
        save_file (str): File name for the saved dataset.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Convert dataset to a list of dictionaries
    data_list = [
        {"input_ids": input_ids.tolist(), "query": query}
        for input_ids, query in zip(dataset.dataframe["input_ids"], dataset.dataframe["query"])
    ]
    
    # Save as a JSON file
    with open(os.path.join(save_dir, save_file), "w") as f:
        json.dump(data_list, f, indent=4)
    
    logging.info(f"Processed dataset saved to {os.path.join(save_dir, save_file)}")

def build_dataset(
    dataset_name: str = "stanfordnlp/imdb",
    input_min_text_length: int = 2,
    input_max_text_length: int = 8,
    min_review_length: int = 200,
    tokenizer_name: str = "gpt2",
):
    """
    Full pipeline to build a tokenized dataset with filters and tokenization.
    
    Args:
        dataset_name (str): Name of the dataset to load.
        input_min_text_length (int): Minimum tokenized text length.
        input_max_text_length (int): Maximum tokenized text length.
        min_review_length (int): Minimum character length for reviews.
        tokenizer_name (str): Name of the tokenizer model to use.
    
    Returns:
        torch.utils.data.Dataset: Tokenized dataset ready for model training.
    """
    # Load and filter dataset
    df = load_and_filter_dataset(dataset_name, min_review_length)

    # Initialize tokenizer
    logging.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    df = tokenize_dataset(df, tokenizer, input_min_text_length, input_max_text_length)

    # Convert to PyTorch Dataset format
    logging.info("Converting DataFrame to PyTorch Dataset.")
    dataset = CustomDataset(df)
    logging.info("Dataset building complete.")
    return dataset

# ---------------------------
# CLI and Main Execution
# ---------------------------
if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="Process and tokenize dataset for RLHF.")
    parser.add_argument("--dataset_name", type=str, default="stanfordnlp/imdb", help="Dataset name.")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer model name.")
    parser.add_argument("--min_review_length", type=int, default=200, help="Minimum character length for reviews.")
    parser.add_argument("--input_min_text_length", type=int, default=2, help="Minimum tokenized text length.")
    parser.add_argument("--input_max_text_length", type=int, default=8, help="Maximum tokenized text length.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
    parser.add_argument("--save_dir", type=str, default="processed_data", help="Directory to save the tokenized dataset.")
    parser.add_argument("--save_file", type=str, default="tokenized_data.json", help="File name for the tokenized dataset.")
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_dir=args.log_dir)

    # Build dataset
    dataset = build_dataset(
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name,
        min_review_length=args.min_review_length,
        input_min_text_length=args.input_min_text_length,
        input_max_text_length=args.input_max_text_length,
    )

    # Save the dataset to a file
    save_dataset_to_file(dataset, save_dir=args.save_dir, save_file=args.save_file)

    # Log an example sample
    logging.info(f"Example tokenized sample: {dataset[0]}")
