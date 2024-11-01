import os
import logging
import requests
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

def create_directory(directory):
    """Creates a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")
    else:
        logging.info(f"Directory already exists: {directory}")

def download_data(url, output_path="data.csv"):
    """
    Downloads data from a URL, extracts it if needed, and ensures it is in CSV format.

    Parameters:
    - url (str): URL of the file to download.
    - output_path (str): Path where the final CSV file will be saved. Defaults to 'data.csv'.

    Returns:
    - DataFrame or None: DataFrame of the data if successful, otherwise None.
    """
    # Create the finetune_data directory
    create_directory("finetune_data")
    
    # Full path for the output CSV file
    output_path = os.path.join("finetune_data", output_path)

    # Download the file if it doesn't already exist
    if not os.path.exists(output_path):
        try:
            logging.info(f"Downloading data from {url}...")
            response = requests.get(url)
            response.raise_for_status()
            zip_path = "downloaded_data.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded data successfully. Saved to {zip_path}")
        except Exception as e:
            logging.error(f"Failed to download data. Error: {e}")
            return None

        # Extract and process the downloaded file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("finetune_data")
                extracted_files = zip_ref.namelist()
            os.remove(zip_path)  # Remove the zip file after extraction
            logging.info(f"Extracted files: {extracted_files}")

            # Only process CSV files
            for file_name in extracted_files:
                if not file_name.endswith('.csv'):
                    logging.warning(f"Skipping unsupported file format: {file_name}")
                    continue

                file_path = os.path.join("finetune_data", file_name)
                logging.info(f"Processing file: {file_name}")

                # Load data as CSV
                try:
                    data = pd.read_csv(file_path)
                    logging.info(f"Loaded data from {file_path} with shape: {data.shape}")
                    
                    # Save data to CSV format if loaded successfully
                    data.to_csv(output_path, index=False)
                    logging.info(f"Data saved to {output_path}")
                    return data
                
                except Exception as e:
                    logging.error(f"Error loading data from {file_path}: {e}")

            logging.error("No suitable CSV file found in the extracted archive.")
            return None

        except Exception as e:
            logging.error(f"Error extracting or processing file: {e}")
            return None

    else:
        logging.info(f"Data file '{output_path}' already exists. Loading data.")
        return pd.read_csv(output_path)

def preprocess_data(data):
    # Map the labels to 0 and 1
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    logging.info(f"Mapped labels to binary values. Unique labels: {data['label'].unique()}")
    
    # Split the data into train, validation, and test sets
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=123, stratify=data["label"])
    validation_data, test_data = train_test_split(temp_data, test_size=0.6667, random_state=123, stratify=temp_data["label"])

    # Save the splits to CSV files with both inputs and targets
    train_data.to_csv("finetune_data/train_data.csv", index=False)
    validation_data.to_csv("finetune_data/validation_data.csv", index=False)
    test_data.to_csv("finetune_data/test_data.csv", index=False)

    # Display the shape of the splits
    logging.info(f"Train shape: {train_data.shape}")
    logging.info(f"Validation shape: {validation_data.shape}")
    logging.info(f"Test shape: {test_data.shape}")

    return train_data, validation_data, test_data

# Main execution
if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    output_path = "sms_spam_collection.csv"
    sms_data = download_data(url, output_path)

    if sms_data is not None:
        logging.info("Data loaded successfully. Displaying head of the data:")
        print(sms_data.head())
        train_data, validation_data, test_data = preprocess_data(sms_data)
    else:
        logging.error("Failed to load data.")
