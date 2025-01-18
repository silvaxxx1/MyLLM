import os
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


from dataloader import GPTDataset, GPTDataLoader  # Make sure to change 'your_module' to the actual module name where GPTDataset and GPTDataLoader are defined

class TestGPTDataset(unittest.TestCase):
    
    def setUp(self):
        """Set up a temporary binary data file for testing."""
        self.test_file = 'test_train_ids.bin'
        self.max_len = 64
        self.stride = 32
        self.batch_size = 8

        # Create a sample dataset and save it as a binary file
        sample_data = np.random.randint(0, 100, size=(1000,), dtype=np.uint16)  # Random integers for testing
        sample_data.tofile(self.test_file)  # Save to binary file

    def tearDown(self):
        """Remove the temporary binary data file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_gpt_dataset_length(self):
        """Test the length of the dataset."""
        dataset = GPTDataset(self.test_file, self.max_len, self.stride)
        # Correct the expected length calculation
        expected_length = (len(np.fromfile(self.test_file, dtype=np.uint16)) - self.max_len) // self.stride + 1
        self.assertEqual(len(dataset), expected_length)

    def test_gpt_dataset_item(self):
        """Test the input and target item retrieval from the dataset."""
        dataset = GPTDataset(self.test_file, self.max_len, self.stride)
        input_ids, target_ids = dataset[0]

        self.assertEqual(input_ids.shape, torch.Size([self.max_len]))
        self.assertEqual(target_ids.shape, torch.Size([self.max_len]))

        # Verify the target is the input shifted by one
        expected_target = input_ids[1:]  # Shift input and take the first max_len - 1 elements
        self.assertTrue(torch.equal(target_ids[:-1], expected_target))  # Compare without the last element

    def test_gpt_dataloader(self):
        """Test the DataLoader for the dataset."""
        dataloader = GPTDataLoader(self.test_file, self.max_len, self.stride, self.batch_size)
        for input_batch, target_batch in dataloader:
            self.assertEqual(input_batch.shape[0], self.batch_size)  # Check batch size
            self.assertEqual(input_batch.shape[1], self.max_len)  # Check sequence length
            self.assertEqual(target_batch.shape[0], self.batch_size)  # Check target batch size
            self.assertEqual(target_batch.shape[1], self.max_len)  # Check target sequence length
            break  # Only check the first batch for this test

if __name__ == '__main__':
    unittest.main()