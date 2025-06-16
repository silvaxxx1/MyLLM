# 🧠 MyLLM: Interactive Notebooks

## *From Data to Aligned Models — A Structured, Hands-On Learning Journey*

<div align="center">
  <img src="images/notepic.jpeg" width="700" alt="LLM Development Roadmap" />
  <br/>
  <em>“Master large language model development through incremental, hands-on experimentation”</em>
</div>

---

## 🌟 Why These Notebooks?

| **Key Aspect**             | **What You'll Gain**                                                      |
| -------------------------- | ------------------------------------------------------------------------- |
| **First Principles**       | Build deep understanding of NLP & transformers by coding from scratch     |
| **Modular Design**         | Develop reusable components integrated into the core framework            |
| **Progressive Learning**   | Systematic flow: data prep → attention → training → alignment → inference |
| **Research to Production** | Bridge your experimental code with production-ready modules in `/modules` |

---

## 🗂 Learning Pathway — Phase by Phase

### Phase 0: Introduction

| Notebook                                         | Status   | Focus                     |
| ------------------------------------------------ | -------- | ------------------------- |
| [0.0.WELCOME.ipynb](notebooks/0.0.WELCOME.ipynb) | ✅ Stable | Overview, setup & tooling |

### Phase 1: Data Foundations 🔍

| Notebook                                             | Status    | Focus                     | Prerequisites   |
| ---------------------------------------------------- | --------- | ------------------------- | --------------- |
| [1.1.DATA.ipynb](notebooks/1.1.DATA.ipynb)           | ✅ Stable  | Text cleaning & splitting | Python basics   |
| [1.2.Tokenizer.ipynb](notebooks/1.2.Tokenizer.ipynb) | 🚧 Active | Byte-level BPE tokenizer  | Regex knowledge |

### Phase 2: Attention Mechanisms 🤖

| Notebook                                                        | Hardware | Key Concept                                  |
| --------------------------------------------------------------- | -------- | -------------------------------------------- |
| [2.1.ATTENTION.ipynb](notebooks/2.1.ATTENTION.ipynb)            | CPU      | Scaled dot-product attention fundamentals    |
| [2.2.More\_ATTENTION.ipynb](notebooks/2.2.More_ATTENTION.ipynb) | GPU      | FlashAttention v2 optimizations              |
| [2.3.GPT.ipynb](notebooks/2.3.GPT.ipynb)                        | GPU      | Transformer architecture from scratch        |
| [2.4.Llama3.ipynb](notebooks/2.4.Llama3.ipynb)                  | GPU      | RoPE positional encoding & SwiGLU activation |

### Phase 3: Model Training 🏢

| Notebook                                              | Status      | Focus                            |
| ----------------------------------------------------- | ----------- | -------------------------------- |
| [3.1.TRAIN.ipynb](notebooks/3.1.TRAIN.ipynb)          | ✅ Stable    | Standard training loops          |
| [3.2.TRAIN\_Pro.ipynb](notebooks/3.2.TRAIN_Pro.ipynb) | 🚧 Advanced | Efficient fine-tuning techniques |

### Phase 4: Supervised Fine-Tuning (SFT) ⚙️

| Notebook                                                                               | Focus                                  |
| -------------------------------------------------------------------------------------- | -------------------------------------- |
| [4.1.SFT\_Text\_Classification.ipynb](notebooks/4.1.SFT_Text_Classification.ipynb)     | Text classification tasks              |
| [4.2.SFT\_Instruction\_Following.ipynb](notebooks/4.2.SFT_Instruction_Following.ipynb) | Instruction tuning                     |
| [4.3.SFT\_PEFT.ipynb](notebooks/4.3.SFT_PEFT.ipynb)                                    | Parameter-efficient fine-tuning (PEFT) |

### Phase 5: Reinforcement Learning from Human Feedback (RLHF) 💪

| Notebook                                            | Focus                          |
| --------------------------------------------------- | ------------------------------ |
| [5.1.RLHF\_PPO.ipynb](notebooks/5.1.RLHF_PPO.ipynb) | Proximal Policy Optimization   |
| [5.2.RL\_DPO.ipynb](notebooks/5.2.RL_DPO.ipynb)     | Direct Preference Optimization |

### Phase 6: Inference & Optimization ⚡

| Notebook                                                                               | Focus                                                                                |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| [6.1.INFERENCE\_Text\_Generation.ipynb](notebooks/6.1.INFERENCE_Text_Generation.ipynb) | Text generation inference                                                            |
| [6.2.KV\_Cache.ipynb](notebooks/6.2.KV_Cache.ipynb)                                    | Key-Value cache optimization                                                         |
| [6.3.Quantization\_8bit.ipynb](notebooks/6.3.Quantization_8bit.ipynb)                  | 8-bit quantization from scratch (PyTorch native)                                     |
| [6.4.Quantization\_Guide.ipynb](notebooks/6.4.Quantization_Guide.ipynb)                | Practical quantization techniques & frameworks (bitsandbytes, GPTQ, GGML, ExLlamaV2) |

### Appendices 📚

| Notebook                                                                     | Focus                                    |
| ---------------------------------------------------------------------------- | ---------------------------------------- |
| [Appendix\_A\_GPT\_2\_Llama2.ipynb](notebooks/Appandix_A_GPT_2_Llama2.ipynb) | GPT-2 vs Llama2 architectural comparison |
| [Appendix\_B\_Gradio.ipynb](notebooks/Appandix_B_Gradio.ipynb)               | Model deployment with Gradio UI          |

---

## 🚀 Getting Started

1. **Setup your environment:**

```bash
conda create -n myllm python=3.10 -y
conda activate myllm
conda install -c pytorch -c nvidia pytorch=2.1.2 torchvision cudatoolkit=12.1 -y
pip install -r notebooks/requirements.txt
```

2. **Launch Jupyter Lab:**

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

3. **Follow a recommended learning path:**

* **Basic track:**
  `1.1.DATA.ipynb` → `1.2.Tokenizer.ipynb` → `2.1.ATTENTION.ipynb` → `3.1.TRAIN.ipynb`

* **Advanced track (GPU required):**
  `2.2.More_ATTENTION.ipynb` → `2.4.Llama3.ipynb` → `6.2.KV_Cache.ipynb`

---

