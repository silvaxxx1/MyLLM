# **MyLLM_Recipes: A Scalable Framework for Building and Fine-Tuning LLMs** 🚀

Welcome to **MyLLM_Recipes**, the next evolutionary step in the **MyLLM101** project! This repository is designed to bridge the gap between simple notebook experiments and a scalable, professional-grade framework for working with large language models (LLMs). Think of this as your vanilla version of Hugging Face's Transformers, but with the added advantage of learning every piece of the puzzle as you build it yourself.

<div align="center">
  <img src="./PIP.png" alt="Logo2" width="800" />
</div>

## Project Overview 🌟

**MyLLM_Recipes** is all about constructing a powerful, reusable framework to:

1. **Load Models**: Seamlessly load pretrained models or initialize new ones from scratch.
2. **Train Models**: Efficiently train LLMs using optimized pipelines for single-GPU, multi-GPU, and advanced configurations.
3. **Fine-Tune Models**: Adapt pretrained models for specific tasks using supervised fine-tuning (SFT) and reinforcement learning techniques like PPO and DPO.
4. **Generate Text**: Utilize cutting-edge generation pipelines for interactive applications and tasks.
5. **Deploy Models**: Integrate inference capabilities with tools like Gradio for user-friendly deployments.

This framework will serve as the foundation for the final phase of the **MyLLM101** project, where we’ll build **Meta_Bot**, an AI tutor that can teach others how it was made.

## Features 💡

### **Pipelines for Every Stage**

#### **1. Data Pipeline**:
- Comprehensive tools for data preprocessing, tokenization, and augmentation.
- Supports custom tokenizers and preprocessing workflows for maximum flexibility.

#### **2. Generation Pipeline**:
- Streamlined setup for generating high-quality text outputs.
- Includes options for sampling strategies like top-k, top-p, and beam search.

#### **3. Training Pipelines**:
- **Training from Scratch**:
  1. **Simple Trainer**: Train models on a single GPU with minimal setup.
  2. **Optimized Trainer**: Utilize mixed precision training and gradient accumulation for faster, more efficient training.
  3. **Multi-GPU Training**: Harness the power of distributed training to scale up your model development.

- **Fine-Tuning Pipelines**:
  1. **SFTTrainer**: Fine-tune models for both classification and instruction-based tasks.
  2. **PPOTrainer**: Implement proximal policy optimization for reinforcement learning.
  3. **DPOTrainer**: Optimize decision processes with advanced training methods.

#### **4. Inference and Deployment**:
- Ready-to-use inference pipeline for fast and efficient model predictions.
- Integrates with **Gradio** to create interactive web-based applications for easy deployment.

### **Framework Features**

- **Automatic Device Detection and Setup**:
  - Automatically detects available hardware (CPU, GPU, multi-GPU setups) and configures the environment.

- **Metric Tracking and Visualization**:
  - Built-in support for tracking training metrics and generating visualizations.
  - Compatible with popular tools like TensorBoard and Weights & Biases for seamless monitoring.

- **Systematic Evaluation**:
  - Provides a robust evaluation framework to assess model performance using accuracy, fluency, and relevance metrics.

## Current Status 📋

This repository is at the starting phase, serving as the blueprint for what we aim to build. While the implementation is still in progress, the roadmap below outlines our structured approach:

| **Feature**                        | **Status**     | **Notes**                                  |
| ---------------------------------- | -------------- | ------------------------------------------ |
| Data Pipeline                      | 🛠️ Upcoming   | Preprocessing and tokenization tools.      |
| Generation Pipeline                | 🛠️ Upcoming   | Advanced text generation techniques.       |
| Simple Trainer                     | 🛠️ Upcoming   | Single-GPU training setup.                 |
| Optimized Trainer                  | 🛠️ Upcoming   | Mixed precision and gradient accumulation. |
| Multi-GPU Training Setup           | 🛠️ Upcoming   | Distributed training capabilities.         |
| SFTTrainer                         | 🛠️ Upcoming   | Supervised fine-tuning for LLMs.           |
| PPOTrainer                         | 🛠️ Upcoming   | Reinforcement learning with PPO.           |
| DPOTrainer                         | 🛠️ Upcoming   | Decision process optimization techniques.  |
| Inference Pipeline                 | 🛠️ Upcoming   | Efficient model predictions.               |
| Gradio Deployment                  | 🛠️ Upcoming   | Interactive web-based applications.        |

## Roadmap 🗺️

Here’s how **MyLLM_Recipes** fits into the broader **MyLLM101** project:

1. Transition from notebook-based experiments to a modular, scalable framework.
2. Build and optimize training and fine-tuning pipelines.
3. Implement systematic evaluation and visualization tools.
4. Deploy inference pipelines with Gradio for real-world applications.
5. Use the framework to create **Meta_Bot**, an AI tutor that teaches the very process of its creation.

## Contributing 🤝

This project is open-source, and we welcome contributions! Here’s how you can help:

- Suggest improvements to the pipeline designs.
- Contribute implementations for specific features.
- Report bugs or issues you encounter.

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Join us on this journey to build a scalable, modular framework for LLM development and pave the way for the creation of **Meta_Bot**! 🚀

