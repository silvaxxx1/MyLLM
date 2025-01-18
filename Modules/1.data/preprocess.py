import os
import tiktoken
import numpy as np
import requests
import logging
import argparse

# Configure logging for the script
logging.basicConfig(level=logging.INFO)

# Constants for the data processing
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
INPUT_FILE_NAME = 'data.txt'  # Default name for the input file
SPLIT_RATIO = 0.9  # Ratio for splitting the dataset into training and validation sets

def download_data(url, output_path):
    """
    Download data from the specified URL and save it to the output path.

    Args:
        url (str): The URL to download the data from.
        output_path (str): The local path to save the downloaded file.

    Returns:
        None
    """
    try:
        if not os.path.exists(output_path):
            logging.info(f"Downloading data from {url}...")
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logging.info("Download complete.")
        else:
            logging.info(f"Data file '{output_path}' already exists. Skipping download.")
    except Exception as e:
        logging.error(f"Error downloading the dataset: {e}")

def preprocess_data(input_path, split_ratio):
    """
    Preprocess the raw data and split it into training and validation sets.

    Args:
        input_path (str): The path to the input text file.
        split_ratio (float): The ratio to split the data into training and validation sets.

    Returns:
        None
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()  # Read the entire content of the file

    data_len = len(raw_text)
    logging.info(f'Length of raw text: {data_len}')
    
    # Split the data based on the specified ratio
    training_data = raw_text[:int(data_len * split_ratio)]
    validation_data = raw_text[int(data_len * split_ratio):]
    
    logging.info(f'Length of training data: {len(training_data)}')
    logging.info(f'Length of validation data: {len(validation_data)}')

    # Tokenization using the specified tokenizer
    tok = tiktoken.get_encoding('gpt2')  # Initialize the tokenizer
    train_ids = tok.encode_ordinary(training_data)  # Tokenize training data
    val_ids = tok.encode_ordinary(validation_data)  # Tokenize validation data

    # Convert tokenized data to NumPy arrays for efficient storage
    train_ids = np.array(train_ids, dtype=np.uint16)    
    val_ids = np.array(val_ids, dtype=np.uint16)

    output_dir = os.path.dirname(__file__)  # Get the directory of the current script
    
    # Save the tokenized data to binary files
    train_ids.tofile(os.path.join(output_dir, 'train_ids.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val_ids.bin'))

    logging.info("Preprocessing complete.")

def main():
    """
    Main function to parse arguments and execute the data download and preprocessing.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Preprocess the dataset for the language model.')
    parser.add_argument('--url', type=str, default=DATA_URL, help='URL of the dataset to download')
    parser.add_argument('--input', type=str, default=INPUT_FILE_NAME, help='Name of the input file')
    parser.add_argument('--split', type=float, default=SPLIT_RATIO, help='Ratio for splitting data (default: 0.9)')

    args = parser.parse_args()  # Parse command line arguments

    input_file_path = os.path.join(os.path.dirname(__file__), args.input)  # Construct input file path
    
    # Download and preprocess data
    download_data(args.url, input_file_path)
    preprocess_data(input_file_path, args.split)

if __name__ == "__main__":
    main()  # Run the main function


# How to use this script:
# 1. Save this script as 'preprocess.py' in your desired directory.
# 2. To run the script, open your terminal (or Anaconda Prompt) and navigate to the directory where 'preprocess.py' is located.
# 3. Use the following command to run the script:
#    python preprocess.py
#
# 4. You can also specify a custom dataset URL, input file name, and split ratio using command-line arguments. 
#    For example:
#    python preprocess.py --url <YOUR_CUSTOM_URL> --input <YOUR_CUSTOM_FILENAME> --split <YOUR_SPLIT_RATIO>
#
# 5. After running the script, it will download the dataset (if not already present), preprocess the data, and save the
#    training and validation data as binary files 'train_ids.bin' and 'val_ids.bin' in the same directory.
