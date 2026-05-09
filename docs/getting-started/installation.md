# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+

## Install from GitHub (recommended)

```bash
pip install git+https://github.com/silvaxxx1/MyLLM.git
```

```bash
# or with uv
uv add git+https://github.com/silvaxxx1/MyLLM.git
```

Always installs the latest `main` branch. Pin to a specific commit or tag:
```bash
pip install git+https://github.com/silvaxxx1/MyLLM.git@v0.1.0
```

## Editable install (local development)

```bash
git clone https://github.com/silvaxxx1/MyLLM.git
cd MyLLM

# with uv (preferred)
uv sync
uv pip install -e .

# or plain pip
pip install -e .
```

## Google Colab

Paste in the first cell of any notebook, then **Runtime → Restart runtime**:

```python
import subprocess, sys, os

def _is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if _is_colab():
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install',
         'git+https://github.com/silvaxxx1/MyLLM.git'],
        capture_output=True, text=True
    )
    print('Done!' if r.returncode == 0 else r.stderr[-2000:])
else:
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    has_uv = subprocess.run(['which', 'uv'], capture_output=True).returncode == 0
    cmd = ['uv', 'pip', 'install', '-e', root] if has_uv else [sys.executable, '-m', 'pip', 'install', '-e', root]
    subprocess.run(cmd)
```

## Optional dependency groups

```bash
# Training extras (wandb, accelerate, deepspeed)
pip install "myllm[train] @ git+https://github.com/silvaxxx1/MyLLM.git"

# Inference extras (matplotlib, pandas, seaborn for benchmarking)
pip install "myllm[inference] @ git+https://github.com/silvaxxx1/MyLLM.git"

# Development (pytest)
pip install "myllm[dev] @ git+https://github.com/silvaxxx1/MyLLM.git"

# Everything
pip install "myllm[all] @ git+https://github.com/silvaxxx1/MyLLM.git"
```

## Verify installation

```python
import myllm
print(myllm.__version__)        # 0.1.0
print(myllm.__all__)            # full list of public exports

# CLI
# python -m myllm version
# python -m myllm models
```

## Core dependencies (auto-installed)

| Package | Purpose |
|---------|---------|
| `torch>=2.0` | Core ML framework |
| `tiktoken>=0.5` | GPT-2/3/4 tokenizer |
| `sentencepiece>=0.1.99` | LLaMA-2 tokenizer |
| `safetensors>=0.4` | Weight file format |
| `transformers>=4.40` | HuggingFace tokenizers (used in demos) |
| `rich>=13.0` | Progress bars and training summaries |
| `pyyaml>=6.0` | YAML config support |
| `psutil>=5.9` | Memory profiling |
