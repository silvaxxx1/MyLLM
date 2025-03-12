import torch
import matplotlib.pyplot as plt
import numpy as np

# Configurations
emb_dim = 128  # Example embedding size
num_heads = 8  # Example number of heads
context_length = 4096  # Original context length
theta = 10000.0  # Standard LLaMA 2 theta

head_dim = emb_dim // num_heads
theta_num = torch.arange(0, head_dim // 2).float()

# Compute inverse frequencies (RoPE scaling)
inv_freq = 1.0 / (theta ** (2 * theta_num / head_dim))

# Compute frequency matrix for all positions
m = torch.arange(context_length)
freq = torch.outer(m, inv_freq)

# Convert to numpy for plotting
freq_np = freq.numpy()

# Plot standard RoPE frequencies
plt.figure(figsize=(10, 5))
plt.imshow(freq_np.T, aspect="auto", cmap="viridis")
plt.colorbar(label="Frequency Value")
plt.xlabel("Position Index")
plt.ylabel("Frequency Dimension")
plt.title("Standard RoPE Frequencies (Theta=10K)")
plt.show()

# LLaMA 3 Frequency Scaling Parameters
freq_config = {
    "original_context_length": 4096,
    "low_freq_factor": 2.0,
    "high_freq_factor": 0.5,
    "factor": 1.5
}

# Compute wavelength
wavelen = 2 * np.pi / inv_freq.numpy()

# Define low and high frequency cutoffs
low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

# Apply LLaMA 3 Scaling
inv_freq_llama = np.where(
    wavelen > low_freq_wavelen, inv_freq.numpy() / freq_config["factor"], inv_freq.numpy()
)

# Smooth interpolation factor
smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
    freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
)

# Apply smooth interpolation in the middle range
smoothed_inv_freq = (
    (1 - smooth_factor) * (inv_freq.numpy() / freq_config["factor"]) + smooth_factor * inv_freq.numpy()
)

is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
inv_freq_llama = np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

# Compute new frequency matrix
freq_llama = np.outer(m.numpy(), inv_freq_llama)

# Plot LLaMA 3 Frequencies
plt.figure(figsize=(10, 5))
plt.imshow(freq_llama.T, aspect="auto", cmap="magma")
plt.colorbar(label="Frequency Value")
plt.xlabel("Position Index")
plt.ylabel("Frequency Dimension")
plt.title("LLaMA 3 Scaled Frequencies (Smooth RoPE)")
plt.show()

plt.figure(figsize=(10, 5))

# Select a few frequency dimensions to compare
valid_indices = np.linspace(0, freq_np.shape[1] - 1, 4, dtype=int)  # Ensures valid indices

for dim in valid_indices:
    plt.plot(m.numpy(), freq_np[:, dim], linestyle="dashed", label=f"RoPE {dim} (Theta=10K)")
    plt.plot(m.numpy(), freq_llama[:, dim], linestyle="solid", label=f"LLaMA 3 {dim} (Scaled)")

plt.xlabel("Position Index")
plt.ylabel("Frequency Value")
plt.title("Comparison of Fixed vs. Scaled RoPE Frequencies")
plt.legend()
plt.show()