import time
import torch
import matplotlib.pyplot as plt
import logging

# Assuming the models are already imported
from MHA import MultiheadAttention, MHACombinedQKV
from MQA import MultiQueryAttention
from FLASH import MHAFlashAttention, MHAPyTorchClass
from GQA import GroupQueryAttention

# Set up logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.StreamHandler(),  # Stream to console
    logging.FileHandler('benchmark_log.txt', mode='w')  # Write to file
])

def benchmark_models(models, embeddings):
    """
    Benchmark the execution time of multiple models and plot the results.
    
    Parameters:
    - models (dict): A dictionary where keys are model names and values are model instances.
    - embeddings (torch.Tensor): The input tensor (embeddings) to run through the models.
    """
    # Initialize a dictionary to store the timing results
    timing_results = {}

    # Measure the time for each model's forward pass
    for model_name, model in models.items():
        # Ensure the model is on the right device
        model = model.to(embeddings.device)

        # Log model information
        logging.info(f"Running benchmark for model: {model_name}")

        # Timing the forward pass
        start_time = time.time()
        _ = model(embeddings)
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        timing_results[model_name] = elapsed_time

        # Log execution time
        logging.info(f"{model_name} - Time taken: {elapsed_time:.6f} seconds")

    # Plotting the timing results
    logging.info("Benchmarking complete. Plotting results.")
    plt.figure(figsize=(10, 6))
    plt.bar(timing_results.keys(), timing_results.values(), color='skyblue')
    plt.xlabel("Model")
    plt.ylabel("Time (seconds)")
    plt.title("Model Execution Time Comparison")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# Set up the device based on availability of CUDA
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Running on: {device} (PyTorch version: {torch.__version__})")

# Input tensor (embeddings) dimensions
batch_size = 8
context_len = 1024
embed_dim = 768
embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)

# Log initial information about the benchmark
logging.info(f"Batch size: {batch_size}, Context length: {context_len}, Embedding dimension: {embed_dim}")

# Initialize models
mha = MultiheadAttention(d_in=embed_dim, d_out=embed_dim, num_heads=8, context_length=context_len).to(device)
mha_combined_qkv = MHACombinedQKV(d_in=embed_dim, d_out=embed_dim, num_heads=8, context_length=context_len).to(device)
multi_query_attention = MultiQueryAttention(d_in=embed_dim, d_out=embed_dim, num_heads=8, context_length=context_len).to(device)
group_query_attention = GroupQueryAttention(d_in=embed_dim, d_out=embed_dim, num_heads=8, context_length=context_len, num_groups=4).to(device)
mha_flash_attention = MHAFlashAttention(d_in=embed_dim, d_out=embed_dim, num_heads=8, context_length=context_len).to(device)
mha_pytorch_class = MHAPyTorchClass(d_in=embed_dim, d_out=embed_dim, num_heads=8, context_length=context_len).to(device)

# Define your models here (make sure they are properly initialized)
models = {
    "MultiheadAttention": mha,
    "MHACombinedQKV": mha_combined_qkv,
    "MultiQueryAttention": multi_query_attention,
    "GroupQueryAttention": group_query_attention,
    "MHAFlashAttention": mha_flash_attention,
    "MHAPyTorchClass": mha_pytorch_class
}

# Call the benchmark function
benchmark_models(models, embeddings)
