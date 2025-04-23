from config import Config 

# Example usage
# Test initialization
config = Config.from_name("gpt2-xl")
print(f"Config name: {config.name}")
print(f"n_embd: {config.n_embd}")
print(f"n_layer: {config.n_layer}")
print(f"n_head: {config.n_head}")
print(f"mlp_class_name: {config.mlp_class_name}")

# Test memory estimation
memory_estimation = config.estimate_memory(batch_size=1)
print("Memory Estimation (in GB):")
for key, value in memory_estimation.items():
    print(f"{key}: {value:.6f}")

# Test save/load functionality
config.save("test_config.json")
loaded_config = Config.load("test_config.json")
print(f"Loaded config name: {loaded_config.name}")

# Test accessing the MLP and Norm classes
mlp_class = config.mlp_class
print(f"MLP Class: {mlp_class}")
norm_class = config.norm_class
print(f"Norm Class: {norm_class}")