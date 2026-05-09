# TrainerEngine

**File:** `myllm/Train/Engine/trainer_engine.py`

Generic, composable training loop decoupled from trainer logic.
Wires together trainer, accelerator, optimizer, scheduler, checkpointer, and callbacks.

---

## When to use

`SFTTrainer` and `PretrainTrainer` have self-contained `train()` methods.
Use `TrainerEngine` when you want:
- Plug-and-play accelerator swapping (single GPU → DDP → DeepSpeed)
- Custom callbacks without modifying trainer code
- A reusable loop across multiple trainer types

---

## Constructor

```python
TrainerEngine(
    trainer,                   # any BaseTrainer subclass
    accelerator,               # BaseAccelerator instance
    optimizer_manager,         # OptimizerManager
    scheduler_manager=None,    # SchedulerManager (optional)
    checkpoint_manager=None,   # CheckpointManager (optional)
    callbacks=[],              # list of Callback instances
    config={},                 # dict with "num_epochs", "gradient_clip", etc.
)
```

---

## Usage

```python
from myllm.Train.Engine.trainer_engine import TrainerEngine
from myllm.Train.Engine.accelerator.single_gpu import SingleGPUAccelerator
from myllm.Train.Engine.optimizer import OptimizerManager

engine = TrainerEngine(
    trainer=my_sft_trainer,
    accelerator=SingleGPUAccelerator(),
    optimizer_manager=OptimizerManager(lr=3e-4, weight_decay=0.1),
    callbacks=[LoggingCallback(), EarlyStoppingCallback(patience=3)],
    config={"num_epochs": 3, "gradient_clip": 1.0},
)

engine.setup()
engine.train()
```

---

## `setup()`

1. `accelerator.setup()` — prepare device, distributed context
2. `accelerator.prepare_model(trainer.model)` — wrap for distributed training
3. `optimizer_manager.setup_optimizer()` → `accelerator.prepare_optimizer(opt)`
4. `scheduler_manager.setup_scheduler()` (if provided)
5. `trainer.on_setup(engine)` (if the trainer has this hook)

## `train()`

```
on_train_start callbacks
for epoch in range(num_epochs):
    on_epoch_start callbacks
    model.train()
    for step, batch in enumerate(trainer.train_dataloader):
        batch = trainer.batch_to_device(batch, device)
        loss = trainer.train_step(batch)
        accelerator.backward(loss, optimizer)
        gradient_clip (if configured)
        optimizer.step() + zero_grad()
        scheduler.step() (if any)
        on_step_end callbacks
        checkpoint (if configured)
    on_epoch_end callbacks
on_train_end callbacks
```

---

## Callback hooks

| Hook | When called |
|------|------------|
| `on_train_start(engine)` | Before first epoch |
| `on_epoch_start(engine, epoch)` | Before each epoch |
| `on_step_end(engine, step, loss)` | After each optimizer step |
| `on_epoch_end(engine, epoch)` | After each epoch |
| `on_train_end(engine)` | After all epochs |

Implement by subclassing `Callback` (see [Callbacks](callbacks.md)).
