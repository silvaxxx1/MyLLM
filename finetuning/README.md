# MyLLM : Fine-Tuning

The `MyLLM/fine-tuning` directory contains scripts and resources for fine-tuning language models using custom datasets. This allows you to adapt pre-trained models to specific tasks, enhancing their performance on domain-specific text or tasks such as spam detection and instruction following.

<p align="center">
    <img src="finetune.png" alt="My Image" />
</p>

## Directory Structure

```
MyLLM/
└── fine-tuning/
    ├── GPT2_124M_SPAM/
    │   ├── eval.py                # Script for evaluating the fine-tuned model
    │   ├── finetune.py            # Script for training the model with a custom dataset
    │   ├── model.py               # Definition of the language model architecture
    │   ├── data.py                # Custom dataset class for loading and preprocessing data
    │   ├── utility.py             # Additional utility functions for processing data
    │               
    ├── GPT2_335M_IF/
    │   ├── eval.py                # Script for evaluating the fine-tuned model
    │   ├── finetune.py            # Script for training the model with a custom dataset
    │   ├── model.py               # Definition of the language model architecture
    │   ├── data.py                # Custom dataset class for loading and preprocessing data
    │   ├── utils.py               # Utility functions for model configuration
    │   └── finetune.utils.py      # Additional utility functions for training 
    │   
    └── README.md                  # this file
```

## Fine-Tuning Projects

### 1. Fine-Tuning GPT-2 124M for Spam Detection

The **GPT2_124M_SPAM** project focuses on fine-tuning the smaller GPT-2 model (124M parameters) for spam detection. The model is trained to classify text as spam or not spam, improving accuracy through custom training.

#### Scripts Overview

- **`eval.py`**: Evaluates the performance of the fine-tuned model on a test dataset.
- **`finetune.py`**: Manages the training process, including data loaders and the training loop.
- **`model.py`**: Defines the architecture of the language model used for fine-tuning.
- **`data.py`**: Contains the custom dataset class for loading and preprocessing training and validation data.
- **`utility.py`**: Includes additional utility functions for data processing.

#### Usage:

1. **Prepare Your Dataset**: Ensure your dataset is in the correct format (e.g., CSV, JSON). Update `data.py` if necessary.

2. **Fine-Tune the Model**:
   Execute the training script with the desired parameters:
   ```bash
   python finetune.py --train_data path/to/spam_train_data --val_data path/to/spam_val_data --model_path path/to/save_spam_model --num_epochs 10 --batch_size 32
   ```

3. **Evaluate the Model**:
   After training, evaluate the model using:
   ```bash
   python eval.py --model_path path/to/saved_spam_model --test_data path/to/spam_test_data --batch_size 32
   ```

4. **Interactive Classification**: After evaluation, enter text for classification directly in the terminal.

---

### 2. Fine-Tuning GPT-2 335M for Instruction Following

The **GPT2_335M_IF** project expands on the previous work by incorporating a larger GPT-2 model (335M parameters) specifically designed for instruction following tasks. This enables the model to understand and generate responses based on user instructions, providing improved interaction and adaptability for various applications.

#### Scripts Overview

- **`eval.py`**: Evaluates the performance of the fine-tuned model on a test dataset.
- **`finetune.py`**: Manages the training process, including data loaders and the training loop.
- **`model.py`**: Defines the architecture of the language model used for fine-tuning.
- **`data.py`**: Contains the custom dataset class for loading and preprocessing training and validation data.
- **`utils.py`**: Includes utility functions for model configuration.
- **`finetune.utils.py`**: Contains additional utility functions for data processing.

#### Usage:

1. **Prepare Your Dataset**: Ensure your dataset is in the appropriate format (e.g., CSV, JSON) for instruction following. Update `data.py` if necessary.

2. **Fine-Tune the Model**:
   Execute the training script with the desired parameters:
   ```bash
   python finetune.py --train_data path/to/instruction_train_data --val_data path/to/instruction_val_data --model_path path/to/save_instruction_model --num_epochs 10 --batch_size 32
   ```

3. **Evaluate the Model**:
   After training, evaluate the model using:
   ```bash
   python eval.py --model_path path/to/saved_instruction_model --test_data path/to/instruction_test_data --batch_size 32
   ```

4. **Interactive Classification**: After evaluation, you can enter text for classification directly in the terminal.

---

### 3. Fine-Tuning GPT-2 XL on the Alpaca Dataset

The **GPT2_XL_ALPACA** project will focus on fine-tuning the largest variant of GPT-2 (GPT-2 XL) using the Alpaca dataset, which contains over 50,000 samples designed for instruction-following tasks. This project will utilize advanced training techniques, including Low-Rank Adaptation (LoRA), Quantized LoRA (QLoRA), and Direct Preference Optimization (DPO) to optimize performance and efficiency during training.

#### Upcoming Features

- **LoRA**: A technique that enables efficient fine-tuning by adding low-rank matrices to the pre-trained model's weights.
- **QLoRA**: An extension of LoRA that utilizes quantization to further reduce the memory footprint and enhance training speed.
- **DPO**: Direct Preference Optimization will be employed to refine the model's responses based on user feedback and preferences.

#### Usage:

Details on preparing the dataset, fine-tuning the model, and evaluating its performance will be provided in the forthcoming README for the **GPT2_XL_ALPACA** directory.

---

## Conclusion

This directory provides the necessary scripts and functionality for fine-tuning both GPT-2 124M for spam detection and GPT-2 335M for instruction following. The addition of **GPT2_XL_ALPACA** will further enhance the capabilities of the project, allowing for advanced fine-tuning with state-of-the-art techniques. By adapting these models to your specific datasets, you can enhance their performance for your particular tasks.