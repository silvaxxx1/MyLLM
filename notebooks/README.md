# 🧠 **MyLLM: Notebooks**  
## *From Data to Aligned Models — A Structured Learning Journey*

<div align="center">
  <img src="images/notepic.jpeg" width="700" alt="LLM Development Roadmap">
  <br>
  <em>"Master LLM development through incremental, hands-on experimentation"</em>
</div>

---

## 🌟 **Why These Notebooks?**
| **Key Aspect**         | **What You'll Achieve**                                                                 |
|------------------------|-----------------------------------------------------------------------------------------|
| **First Principles**   | Deep understanding of core NLP concepts through implementation                         |
| **Modular Design**     | Build reusable components that feed directly into the main codebase                     |
| **Progressive Path**   | Systematic progression from data prep → model architecture → alignment                  |
| **Research to Prod**   | Bridge between experimental notebooks and production-grade code in `/modules`           |

---

## 🗀 **Learning Pathway**

### **Phase 0: Introduction**
| Notebook | Status | Focus |
|----------|--------|-------|
| [0.0.WELCOME](notebooks/0.0.WELCOME.ipynb) | ✅ Stable | Overview & Setup |

### **Phase 1: Data Foundations** 🔍
| Notebook | Status | Focus | Prerequisites |
|----------|--------|-------|---------------|
| [1.1.DATA](notebooks/1.1.DATA.ipynb) | ✅ Stable | Text cleaning & splitting | Python basics |
| [1.2.Tokenizer](notebooks/1.2.Tokenizer.ipynb) | 🚧 Active | Byte-level BPE | Regex experience |

### **Phase 2: Attention Mechanisms** 🤖
| Notebook | Hardware | Key Innovation |
|----------|----------|----------------|
| [2.1.ATTENTION](notebooks/2.1.ATTENTION.ipynb) | CPU | Scaled dot-product basics |
| [2.2.More_ATTENTION](notebooks/2.2.More_ATTENTION.ipynb) | GPU | FlashAttention v2 |
| [2.3.GPT](notebooks/2.3.GPT.ipynb) | GPU | Transformer architecture |
| [2.4.Llama3](notebooks/2.4.Llama3.ipynb) | GPU | RoPE & SwiGLU |

### **Phase 3: Model Training** 🏢
| Notebook | Status | Focus |
|----------|--------|-------|
| [3.1.TRAIN](notebooks/3.1.TRAIN.ipynb) | ✅ Stable | Standard training |
| [3.2.TRAIN_Pro](notebooks/3.2.TRAIN_Pro.ipynb) | 🚧 Advanced | Efficient fine-tuning |

### **Phase 4: Supervised Fine-Tuning (SFT)** ⚙️
| Notebook | Focus |
|----------|-------|
| [4.1.SFT_Text_Classification](notebooks/4.1.SFT_Text_Classification.ipynb) | Text classification |
| [4.2.SFT_Instruction_Following](notebooks/4.2.SFT_Instruction_Following.ipynb) | Instruction fine-tuning |
| [4.3.SFT_PEFT](notebooks/4.3.SFT_PEFT.ipynb) | Parameter Efficient Fine-Tuning (PEFT) |

### **Phase 5: Reinforcement Learning from Human Feedback (RLHF)** 💪
| Notebook | Focus |
|----------|-------|
| [5.1.RLHF_PPO](notebooks/5.1.RLHF_PPO.ipynb) | Proximal Policy Optimization |
| [5.2.RL_DPO](notebooks/5.2.RL_DPO.ipynb) | Direct Preference Optimization |

### **Phase 6: Inference & Optimization** ⚡
| Notebook | Focus |
|----------|-------|
| [6.1.INFERENCE_Text_Generation](notebooks/6.1.INFERENCE_Text_Generation.ipynb) | Text generation |
| [6.2.KV_Cache](notebooks/6.2.KV_Cache.ipynb) | KV cache optimization |
| [6.3.Quantization_8bit](notebooks/6.3.Quantization_8bit.ipynb) | 8-bit quantization (from scratch) |
| [6.4.Quantization_Guide](notebooks/6.4.Quantization_Guide.ipynb) | Practical guide to quantization techniques and frameworks |

### **Appendices** 📚
| Notebook | Focus |
|----------|-------|
| [Appendix_A_GPT_2_Llama2](notebooks/Appandix_A_GPT_2_Llama2.ipynb) | Comparing GPT-2 & Llama2 |
| [Appendix_B_Gradio](notebooks/Appandix_B_Gradio.ipynb) | Deploying models with Gradio |

---

## 🚀 **Getting Started**

1. **Prerequisites**:
   ```bash
   conda create -n myllm python=3.10
   conda install -c pytorch -c nvidia pytorch=2.1.2 torchvision cudatoolkit=12.1
   pip install -r notebooks/requirements.txt
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
   ```

3. **Start Learning**:
   ```bash
   # Basic path
   1.1_DATA → 1.2_Tokenizer → 2.1_ATTENTION → 3.1_TRAIN

   # Advanced path (GPU required)
   2.2_MORE_ATTENTION → 2.4_Llama3 → 6.2_KV_Cache
   ```

---

## 🌌 **Roadmap 2024**

```python
class NotebookRoadmap:
    Q1 = [
        "📈 AutoML for Hyperparameter Tuning",
        "🌐 Multilingual Tokenizer Expansion",
        "🔍 Interpretability Suite"
    ]
    
    Q2 = [
        "🤖 Autonomous RLHF Pipeline",
        "🔄 Notebook→Colab One-Click Export"
    ]
```

---

### Updates:
In **Phase 6** (Inference & Optimization), the quantization section has been expanded:

- **6.3.Quantization_8bit**: This notebook guides you through **8-bit quantization** from scratch using **native PyTorch**, enabling efficient model optimization with minimal code complexity.
- **6.4.Quantization_Guide**: The second notebook in this section is a **practical guide** to **quantization techniques** and frameworks like **bitsandbytes**, **GPTQ**, **GGML & llama.cpp**, and **ExLlamaV2**, helping you understand **dynamic** vs **static** quantization and implementing them for real-world models.