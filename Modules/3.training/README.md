# MyLLM / Training

This directory implements a flexible training pipeline for various transformer-based models, including **GPT**, **Llama**, and **BERT**, using **PyTorch**. The training process is designed for text generation, language modeling, and other NLP tasks. The pipeline supports both standard and distributed training configurations to scale training across multiple GPUs efficiently.

## Directory Structure

```
training/
│
├── train.py                # Contains the training loop and loss functions for model training.
│
├── train_dist.py           # Distributed training script using DDP with `torchrun` for multi-GPU setups.
│
├── trainer.py              # Main script to initialize and execute the training process. It handles argument parsing and model configuration.
│
├── train_utils.py          # Utility functions for token generation, text conversion, loss visualization, and model saving/loading.
│
├── models/                 # Folder containing GPT, Llama, and BERT model implementations.
└── config/                 # Configuration files for each model's architecture and hyperparameters.
```

## File Descriptions

- **train.py**: This file contains the core functions for computing loss and executing the training loop. It handles training, validation loss computation, and model evaluation.

- **train_dist.py**: The distributed training script using `torch.distributed` and `torchrun` for multi-GPU training. It uses the `DistributedDataParallel` (DDP) module for parallelizing training across multiple GPUs. It supports gradient accumulation, checkpointing, and periodic evaluations during distributed training.

- **trainer.py**: This is the entry point for standard training. It sets up the command-line interface, initializes the model (GPT, Llama, or BERT), loads the data, and starts the training and evaluation procedures. It supports both single-GPU and multi-GPU configurations.

- **train_utils.py**: This module contains helper functions for token generation, text conversion, loss visualization, and saving/loading model checkpoints. It aids in tracking the training progress and making predictions with the trained model.

- **models/**: This folder contains the model architectures for GPT, Llama, and BERT. Each model can be configured and trained based on your needs.

- **config/**: This folder includes configuration files for each model, containing the architecture, hyperparameters, and training parameters tailored for GPT, Llama, and BERT.

## Model Support

This training pipeline supports the following models:

1. **GPT**:
   - Designed for text generation tasks.
   - Can be trained from scratch or fine-tuned on specific data for language modeling.

2. **Llama**:
   - A more efficient transformer model, optimized for high performance with fewer parameters.
   - Supports tasks such as text generation, contextual reasoning, and knowledge retrieval.

3. **BERT**:
   - Primarily used for language understanding tasks, such as classification, token classification, and question answering.
   - Supports both masked language modeling (MLM) and next sentence prediction (NSP).

Each model is implemented in a modular way, and you can select the appropriate configuration and training script depending on your use case.

## Key Features

### 1. **Modular and Flexible**

- The pipeline is designed to work with multiple transformer models (GPT, Llama, and BERT), enabling you to switch between models easily.
- Hyperparameters, architecture configurations, and training parameters are separated into different config files, which makes it easy to adjust settings for each model.

### 2. **Distributed Training Support**

- **train_dist.py** allows you to scale your training across multiple GPUs using **DistributedDataParallel (DDP)** and **torchrun**.
- Gradient accumulation is implemented to handle larger batch sizes when training across multiple GPUs.
- **Checkpointing** is supported to save model states periodically during distributed training, ensuring that training can be resumed if interrupted.

### 3. **Enhanced Training Loops**

- **V1**: Basic training loop for standard evaluation.
- **V2**: More advanced training loop with improvements such as learning rate warm-up, cosine decay, and gradient clipping.
- **Distributed Training**: Optimized for multi-GPU setups, with support for recovery and large batch sizes.

### 4. **Customizable Configuration**

- Each model (GPT, Llama, BERT) has a separate configuration file with model-specific hyperparameters and training parameters.
- You can fine-tune pre-trained models on your own datasets or train from scratch by modifying the config files.

## Command-Line Usage

### V1 Command-Line Usage:
To train using the V1 loop, use the following command:
```bash
python trainer.py --model "<model_name>" \
                  --train_file "<path_to_training_data>" \
                  --val_file "<path_to_validation_data>" \
                  --epochs <number_of_epochs> \
                  --learning_rate <learning_rate_value> \
                  --batch_size <batch_size_value> \
                  --max_len <max_sequence_length> \
                  --stride <stride_value> \
                  --eval_freq <evaluation_frequency> \
                  --eval_iter <number_of_eval_batches> \
                  --start_context "<initial_context>" \
                  --tokenizer "<tokenizer_type>" \
                  --save_model "<path_to_save_model>"
```

- **`<model_name>`**: Specify the model to train (`gpt`, `llama`, or `bert`).
- **`<tokenizer_type>`**: Choose the appropriate tokenizer (`gpt2`, `llama`, or `bert`).

### V2 Command-Line Usage:
To train with the more advanced V2 loop, use the following command:
```bash
python trainer.py --model "<model_name>" \
                  --train_file "<path_to_training_data>" \
                  --val_file "<path_to_validation_data>" \
                  --epochs <number_of_epochs> \
                  --learning_rate <learning_rate_value> \
                  --batch_size <batch_size_value> \
                  --max_len <max_sequence_length> \
                  --stride <stride_value> \
                  --eval_freq <evaluation_frequency> \
                  --eval_iter <number_of_eval_batches> \
                  --start_context "<initial_context>" \
                  --tokenizer "<tokenizer_type>" \
                  --save_model "<path_to_save_model>" \
                  --load_model "<path_to_load_model>"
```

### Distributed Training Command-Line Usage:
To use `train_dist.py` for distributed training, use the following command:
```bash
torchrun --nproc_per_node=<number_of_gpus> train_dist.py \
         --model "<model_name>" \
         --train_file "<path_to_training_data>" \
         --val_file "<path_to_validation_data>" \
         --num_epochs <number_of_epochs> \
         --learning_rate <learning_rate_value> \
         --batch_size <batch_size_value> \
         --max_len <max_sequence_length> \
         --stride <stride_value> \
         --eval_freq <evaluation_frequency> \
         --eval_iter <number_of_eval_batches> \
         --start_context "<initial_context>" \
         --tokenizer "<tokenizer_type>" \
         --save_path "<path_to_save_checkpoints>"
```

- **`<model_name>`**: Specify the model to train (`gpt`, `llama`, or `bert`).
- **`--nproc_per_node`**: The number of GPUs to use for distributed training.

## Example Usage with `torchrun`:

```bash
torchrun --nproc_per_node=4 train_dist.py \
         --model "gpt" \
         --train_file "C:\\path\\to\\your\\train_file.bin" \
         --val_file "C:\\path\\to\\your\\val_file.bin" \
         --num_epochs 10 \
         --learning_rate 1e-4 \
         --batch_size 32 \
         --max_len 512 \
         --stride 256 \
         --eval_freq 100 \
         --eval_iter 10 \
         --start_context "" \
         --tokenizer "gpt2" \
         --save_path "C:\\path\\to\\save_checkpoints"
```

## Notes

- The `--save_path` argument specifies the directory where model checkpoints will be saved after each epoch.
- Distributed training requires setting environment variables like `MASTER_ADDR` and `MASTER_PORT` in a multi-node setting.
- Make sure to configure the environment for multi-GPU setups, particularly with the `torchrun` and `DDP` configuration.
- The training scripts automatically adjust to different models (GPT, Llama, BERT) by setting the appropriate `--model` argument.
- The `train_dist.py` script leverages gradient accumulation to efficiently handle larger batch sizes across multiple GPUs.

## Future Enhancements

- Support for additional transformer-based models (e.g., T5, BART).
- Enhanced multi-node distributed training for very large datasets.
- Integration with model-specific pre-processing pipelines (e.g., for BERT and Llama).

## Contribution

We welcome contributions to this repository! If you'd like to add new features, models, or improvements, please open a pull request. Be sure to follow the contribution guidelines in the repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
