# **MyLLM101: My Journey to Building Meta\_Bot** 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

---

## **About This Project**

This repository captures my personal exploration and hands-on journey into Large Language Models (LLMs). It reflects the process I followed — learning, building, iterating — to deeply understand how LLMs work and how to create my own.

It’s organized into **4 key phases** that represent my evolving understanding and code maturity:

---

## **4 Key Phases of My Learning and Build Process**

### 1. 📓 **Notebook — My Interactive Playground**

This is where I started exploring core LLM concepts interactively.
The notebooks are my “sandbox” for trying out ideas, understanding transformers, and experimenting with tokenization and training loops.

### 2. 🧩 **Modules — From Experiments to Code**

Once I grasped the concepts, I refactored notebook code into clean, modular mini-projects.
This helped me organize the codebase and build reusable components — moving towards more practical, maintainable implementations.

### 3. ⚙️ **MyLLM — Building the Engine**

This is the core framework I built from scratch using PyTorch to replicate the engineering behind modern LLM pipelines.
It’s designed to keep dependencies minimal while allowing me to build, train, fine-tune, and run reinforcement learning on transformer models — really getting into the engineering details.

### 4. 🤖 **MetaBot — Bringing It All Together**

As a final step, I used the MyLLM framework to create `MetaBot`, a chatbot that not only answers questions but can explain how it itself was built.
This meta-level application was a milestone in applying everything I’d learned into a tangible, interactive system.

---

## **Why I Built This**

* To *learn by doing* — not just read or watch, but actually build each part.
* To break down complex LLM engineering into understandable pieces.
* To create a pipeline that’s transparent and fully controlled from first principles.
* To experiment with building intelligent applications based on my own models.

---

## **How to Explore This Repo**

If you want to follow my steps or experiment yourself:

```bash
git clone https://github.com/silvaxxx1/MyLLM101.git
cd MyLLM101
pip install -r requirements.txt
```

* Dive into `notebooks/` to see my initial experiments and ideas.
* Check out `modules/` for cleaner, modularized code versions.
* Explore the `MyLLM/` folder — this is where the core framework lives.
* Run the chatbot:

```bash
python -m metabot.chat --mode basic
```

---

## **Current Progress**

* Core GPT training pipeline implemented and evolving
* Interactive notebooks fully functional
* Modular components organized and tested
* MetaBot chatbot in early working state

---

## **🔍 Preview: MyLLM API — Simple & Powerful**

The **MyLLM** framework offers a clear and intuitive API, inspired by Hugging Face and LitGPT, while keeping everything lightweight and customizable.

```python
from myllm import LLM, SFTTrainer, DPOTTrainer, PPOTrainer, Quantizer

# Load a pretrained or custom LLM model
llm = LLM.load("my_model_checkpoint")

# Generate text with a simple call
output = llm.generate("Once upon a time,", max_length=100)
print(output)

# Supervised Fine-Tuning (SFT)
sft_trainer = SFTTrainer(
    model=llm,
    dataset=my_dataset,
    epochs=3,
    batch_size=8,
    learning_rate=5e-5,
)
sft_trainer.train()

# Direct Preference Optimization (DPO) Trainer for RL fine-tuning
dpo_trainer = DPOTTrainer(
    model=llm,
    dataset=my_dataset,
    epochs=3,
    batch_size=8,
    learning_rate=5e-5,
)
dpo_trainer.train()

# Proximal Policy Optimization (PPO) Trainer for RL
ppo_trainer = PPOTrainer(
    model=llm,
    env=my_env,
    epochs=10,
    batch_size=4,
)
ppo_trainer.train()

# Model Quantization to reduce size and speed up inference
quantizer = Quantizer(model=llm)
quantized_llm = quantizer.apply(precision="int8")

# Save your (fine-tuned or quantized) model
quantized_llm.save("fine_tuned_quantized_model.pt")
```

---

### Why this API?

* **Familiar and simple:** Easy to pick up if you’ve used HF Transformers or LitGPT.
* **Multiple training strategies:** Supports supervised fine-tuning, DPO, PPO, and more.
* **Model optimization:** Built-in quantization for efficient inference.
* **Minimal dependencies:** Pure PyTorch with transparent engineering.
* **Extendable & modular:** Add your own trainers, optimizers, or custom layers easily.

---

## **Inspired By**

This project was inspired by the brilliant work of:

* Umar Jamil — for practical transformer tutorials
* Andrej Karpathy’s nanoGPT — minimalistic and elegant GPT code
* Sebastian Raschka’s LLM Foundations — in-depth theoretical and practical insights

Their resources motivated me to build something that’s both a learning tool and an engineering project.

---

## **License**

MIT License — Feel free to explore, adapt, and learn along.

---

<div align="center">  
  <img src="./META_BOT.jpg" alt="Meta_Bot" width="600" />  
  <br>  
  <em>Join me in this hands-on journey — building, learning, and evolving an LLM from scratch.</em>  
</div>

---
