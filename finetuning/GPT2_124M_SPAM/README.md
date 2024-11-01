# MyLLM : Fine-Tuning

The `MyLLM/fine-tuning` directory contains scripts and resources for fine-tuning a language model using a custom dataset. This allows you to adapt a pre-trained model to your specific tasks, enhancing its performance on domain-specific text or tasks such as sentiment analysis, text classification, or any other natural language processing application.

## Directory Structure

```
MyLLM/
└── fine-tuning/
    ├── eval.py                # Script for evaluating the fine-tuned model
    ├── finetune.py               # Script for training the model with a custom dataset
    ├── model.py               # Definition of the language model architecture
    ├── data.py             # Custom dataset class for loading and preprocessing data
    └── README.md              # This file
```

## Scripts Overview

- **`eval.py`**: This script evaluates the performance of the fine-tuned model on a test dataset. It calculates accuracy and allows interactive input for real-time classification.

- **`finetune.py`**: This script handles the training process for the model. It sets up the data loaders, optimizer, and training loop. You can specify training parameters through command-line arguments.

- **`model.py`**: Contains the architecture definition of the language model used for fine-tuning. You can modify this file to implement custom layers or architectures as needed.

- **`data.py`**: This file defines the custom dataset class that loads and preprocesses your training and validation data. You can modify the data loading and transformation logic here to suit your dataset format.



## Usage

To fine-tune the language model, follow these steps:

1. **Prepare Your Dataset**: Ensure your dataset is in the correct format (e.g., CSV, JSON). Update the `data.py` if necessary to match your data structure.


2. **Fine-Tune the Model**:
   Execute the training script with the desired parameters:
   ```bash
   python train.py --train_data path/to/train_data --val_data path/to/val_data --model_path path/to/save_model --num_epochs 10 --batch_size 32
   ```

3. **Evaluate the Model**:
   After training, evaluate the model using:
   ```bash
   python eval.py --model_path path/to/saved_model --test_data path/to/test_data --batch_size 32
   ```

4. **Interactive Classification**: After evaluation, you can enter text for classification directly in the terminal.

