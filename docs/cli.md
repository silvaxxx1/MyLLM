# CLI Reference

**File:** `myllm/__main__.py`
**Entry point:** `myllm` command (registered in `pyproject.toml`)

---

## Usage

```bash
myllm <command> [args]

# or
python -m myllm <command> [args]
```

---

## Commands

### `version`

Print the installed package version.

```bash
myllm version
# myllm 0.1.0
```

### `models`

List all registered `ModelConfig` names.

```bash
myllm models
# Available model configs:
#   gpt2-small
#   gpt2-medium
#   gpt2-large
#   gpt2-xl
#   llama2-7b
#   llama2-13b
#   llama3-1b
#   llama3-3b
#   llama3-8b
```

### `info <model>`

Show architecture details and memory estimates for a specific model.

```bash
myllm info gpt2-small
# Model : gpt2-small
# Layers: 12  Heads: 12  Embd: 768
# Params: 151.9M
# Memory (fp32): 0.57 GB params + 0.14 GB activations

myllm info llama2-7b
# Model : llama2-7b
# Layers: 32  Heads: 32  Embd: 4096
# Params: 8721.5M
# Memory (fp32): 32.49 GB params + ...
```

---

## Adding new commands

Edit `myllm/__main__.py`:

```python
# Add to sub parsers
new_cmd = sub.add_parser('generate', help='Generate text from prompt')
new_cmd.add_argument('prompt', help='Input prompt')
new_cmd.add_argument('--model', default='gpt2-small')
new_cmd.add_argument('--max-length', type=int, default=50)

# Handle in main()
elif args.cmd == 'generate':
    from myllm import LLM, GenerationConfig
    llm = LLM.from_pretrained(args.model)
    result = llm.generate_text(
        args.prompt,
        GenerationConfig(max_length=args.max_length),
        skip_prompt=True,
    )
    print(result['text'])
```
