import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import requests

# Add the parent directory to sys.path for module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import logging
from unittest import mock
import preprocess  # Make sure this is the correct import for your preprocess script

# Configure logging for the tests
logging.basicConfig(level=logging.INFO)

class TestPreprocessData(unittest.TestCase):
    
    @mock.patch('preprocess.download_data')  # Mock the download_data function
    def test_preprocess_data(self, mock_download):
        # Mocking the download function to skip actual download
        mock_download.return_value = None
        
        input_path = 'C:\\Users\\user\\Documents\\SILVA AI ROADMAP\\MyLLM\\data\\data.txt'  # Change this path as needed
        split_ratio = 0.9
        
        # Call the preprocess_data function
        preprocess.preprocess_data(input_path, split_ratio)
        
        # Define output files based on the original directory
        output_train_file = 'C:\\Users\\user\\Documents\\SILVA AI ROADMAP\\MyLLM\\data\\train_ids.bin'
        output_val_file = 'C:\\Users\\user\\Documents\\SILVA AI ROADMAP\\MyLLM\\data\\val_ids.bin'
        
        # Check if the output files were created
        self.assertTrue(os.path.exists(output_train_file), "Training data file does not exist.")
        self.assertTrue(os.path.exists(output_val_file), "Validation data file does not exist.")
        
        # Clean up created files after the test
        try:
            if os.path.exists(output_train_file):
                os.remove(output_train_file)
        except PermissionError as e:
            logging.warning(f"Could not delete {output_train_file}: {e}")
        
        try:
            if os.path.exists(output_val_file):
                os.remove(output_val_file)
        except PermissionError as e:
            logging.warning(f"Could not delete {output_val_file}: {e}")

if __name__ == '__main__':
    unittest.main()
