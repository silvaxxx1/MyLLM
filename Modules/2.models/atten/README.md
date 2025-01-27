# Attention Model Benchmarking

This repository contains multiple implementations of attention mechanisms, along with a benchmarking script (`benchmark.py`) that compares the execution times of these models. The benchmarking script tests these attention models on either CPU or GPU, logs detailed performance data, and visualizes the results in a comparison chart.

## Models Implemented

The following attention mechanisms are implemented and benchmarked:

1. **Multihead Attention (MHA)**  
2. **MHA with Combined QKV**  
3. **Multi Query Attention (MQA)**  
4. **Group Query Attention (GQA)**  
5. **MHA with Flash Attention**  
6. **MHAPyTorch Class Attention**

## Features

- **Multiple Attention Mechanisms**: Several variations of attention models, each with unique optimizations and performance characteristics.
- **Automatic Device Selection**: The script automatically detects and uses the best available hardware (GPU/CPU).
- **Benchmarking**: Each attention model is benchmarked on a synthetic tensor with a batch size of 8, context length of 1024, and embedding dimension of 768.
- **Logging**: Execution times are logged in both the console and a file (`benchmark_log.txt`).
- **Visualization**: A bar chart is generated to compare the execution times of all tested models.


### Run the Benchmarking Script:

The script automatically detects whether you're using a CPU or GPU and runs the benchmark accordingly:

```bash
python benchmark.py
```

###  View the Results

- **Console**: Execution times for each attention mechanism will be printed in the terminal.
- **Log File**: All benchmarking data will be saved to `benchmark_log.txt`.
- **Plot**: A bar chart will be displayed showing a comparison of the execution times for each model.

### Example Console Output

```
Running on: cuda (PyTorch version: 1.10.0)
Batch size: 8, Context length: 1024, Embedding dimension: 768
Running benchmark for model: MultiheadAttention
MultiheadAttention - Time taken: 0.058247 seconds
Running benchmark for model: MHACombinedQKV
MHACombinedQKV - Time taken: 0.072345 seconds
...
Benchmarking complete. Plotting results.
```

### Example Log File (`benchmark_log.txt`)

```
2025-01-27 12:00:00 - Running on: cuda (PyTorch version: 1.10.0)
2025-01-27 12:00:05 - Running benchmark for model: MultiheadAttention
2025-01-27 12:00:06 - MultiheadAttention - Time taken: 0.058247 seconds
2025-01-27 12:00:08 - Running benchmark for model: MHACombinedQKV
2025-01-27 12:00:09 - MHACombinedQKV - Time taken: 0.072345 seconds
...
2025-01-27 12:05:00 - Benchmarking complete. Plotting results.
```

## Attention Mechanisms Benchmarked

1. **Multihead Attention (MHA)**  
   A standard attention mechanism where multiple attention heads are computed separately and combined.

2. **MHA with Combined QKV**  
   A variant where the Query, Key, and Value projections are combined into a single step for improved efficiency.

3. **Multi Query Attention (MQA)**  
   A modified attention model that allows multiple queries to share the same Key-Value pairs, improving resource utilization.

4. **Group Query Attention (GQA)**  
   An attention mechanism where queries are grouped together, offering improved efficiency in certain scenarios.

5. **MHA with Flash Attention**  
   An attention model that uses hardware-specific optimizations for faster execution, reducing memory usage and improving performance.

6. **MHAPyTorch Class Attention**  
   A custom implementation of multihead attention using PyTorch's built-in functionalities, allowing for easy integration and modification.
