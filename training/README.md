Here's the updated README to include details on `train_dist.py` and how to use `torchrun` for distributed training:

```markdown
# Mini README
# MyLLM / training

This directory implements a training pipeline for a GPT-based language model using PyTorch. The architecture is designed for text generation tasks, leveraging the transformer model's capabilities. The distributed training setup uses `train_dist.py` for efficient scaling across multiple GPUs.

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
```

## File Descriptions

- **train.py**: This file includes the core functions for computing loss and executing the training loop for the model. It manages the training process and handles the evaluation of training and validation losses.

- **train_dist.py**: The distributed training script using `torch.distributed` and `torchrun` for multi-GPU training. It leverages the `DistributedDataParallel` (DDP) module to parallelize training across multiple GPUs. Logging is used to track progress and evaluations.

- **trainer.py**: This is the entry point of the project for standard training. It sets up the command-line interface for running the training process, initializes the model, loads the data, and starts the training and evaluation procedures for both versions (V1 and V2).

- **train_utils.py**: This module contains helper functions, such as converting token IDs to text, generating new text sequences from the model, and visualizing training. It also includes functionality for saving and loading model states.

## Differences Between V1, V2, and Distributed Training

- **V1**: Basic training loop with standard evaluation.
  
- **V2**: A more robust training loop featuring:
  - Learning rate warm-up
  - Cosine decay for learning rate
  - Gradient clipping

- **Distributed Training**: Implements DDP for multi-GPU training using `train_dist.py`, including:
  - Checkpointing for recovery during distributed training
  - Gradient accumulation for large batch sizes
  - Periodic evaluation across distributed processes

## Command-Line Differences

The command-line arguments for V1, V2, and `train_dist.py` differ slightly:

### V1 Command-Line Usage:
```bash
python trainer.py --train_file "<path_to_training_data>" \
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

### V2 Command-Line Usage:
```bash
python trainer.py --train_file "<path_to_training_data>" \
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
To use `train_dist.py` with `torchrun` for distributed training, use the following command:
```bash
torchrun --nproc_per_node=<number_of_gpus> train_dist.py \
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

- **`--nproc_per_node`** specifies the number of GPUs to use.
- Checkpoints are saved periodically in `<save_path>/latest_checkpoint.pt`.

## Example Usage with `torchrun`:

```bash
torchrun --nproc_per_node=4 train_dist.py \
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
- Distributed training requires setting environment variables such as `MASTER_ADDR` and `MASTER_PORT` in a multi-node setting.
- The `train_dist.py` script uses gradient accumulation to handle large batch sizes across multiple GPUs.
```