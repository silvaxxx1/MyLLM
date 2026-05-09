# Accelerators

**Directory:** `myllm/Train/Engine/accelerator/`

Accelerators abstract the training backend — single GPU, DDP, DeepSpeed, FSDP.
All implement the same interface so `TrainerEngine` is backend-agnostic.

---

## Interface (`BaseAccelerator`)

**File:** `myllm/Train/Engine/accelerator/base.py`

```python
class BaseAccelerator:
    def setup(self): ...
    def prepare_model(self, model) -> nn.Module: ...
    def prepare_optimizer(self, optimizer) -> Optimizer: ...
    def backward(self, loss, optimizer): ...
```

---

## Available accelerators

### `SingleGPUAccelerator`
**File:** `single_gpu.py`

Standard single-GPU training using `torch.cuda`. No distributed setup.
Default for most use cases.

```python
from myllm.Train.Engine.accelerator.single_gpu import SingleGPUAccelerator
acc = SingleGPUAccelerator()
```

### `DDPAccelerator`
**File:** `ddp_accelerate.py`

PyTorch native `DistributedDataParallel`. Requires launching with `torchrun`.

```python
from myllm.Train.Engine.accelerator.ddp_accelerate import DDPAccelerator
acc = DDPAccelerator(local_rank=int(os.environ['LOCAL_RANK']))
```

Launch:
```bash
torchrun --nproc_per_node=4 train.py
```

### `HFAccelerator`
**File:** `hf_accerlerate.py`

HuggingFace `accelerate` library backend. Handles DDP, mixed precision,
and gradient accumulation automatically via `accelerate config`.

```python
from myllm.Train.Engine.accelerator.hf_accerlerate import HFAccelerator
acc = HFAccelerator()
```

### `DeepSpeedAccelerator`
**File:** `deepspeed_accerlerate.py`

Microsoft DeepSpeed ZeRO optimisation stages. Requires a DeepSpeed JSON config.

```python
from myllm.Train.Engine.accelerator.deepspeed_accerlerate import DeepSpeedAccelerator
acc = DeepSpeedAccelerator(config_path='ds_config.json')
```

Requires: `pip install deepspeed`

### `FSPDAccelerator`
**File:** `fspd_accelerate.py`

PyTorch Fully Sharded Data Parallel (FSDP). Shards model parameters, gradients,
and optimizer states across GPUs. Enables training very large models.

```python
from myllm.Train.Engine.accelerator.fspd_accelerate import FSPDAccelerator
acc = FSPDAccelerator()
```

---

## Choosing an accelerator

| Scenario | Recommended |
|----------|------------|
| Single GPU | `SingleGPUAccelerator` |
| Multi-GPU, simple | `DDPAccelerator` or `HFAccelerator` |
| Multi-GPU, memory-constrained | `FSPDAccelerator` |
| Very large models (70B+) | `DeepSpeedAccelerator` (ZeRO-3) |
| Existing `accelerate` workflow | `HFAccelerator` |
