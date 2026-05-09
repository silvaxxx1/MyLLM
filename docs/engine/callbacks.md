# Callbacks

**File:** `myllm/Train/Engine/callbacks.py`

Hook system for injecting logic at specific points in the `TrainerEngine` training loop.

---

## Base class

```python
class Callback:
    def on_train_start(self, engine, **kwargs): pass
    def on_epoch_start(self, engine, epoch, **kwargs): pass
    def on_step_end(self, engine, step, loss, **kwargs): pass
    def on_epoch_end(self, engine, epoch, **kwargs): pass
    def on_train_end(self, engine, **kwargs): pass
```

All methods are no-ops by default — override only what you need.

---

## Usage

```python
from myllm.Train.Engine.callbacks import Callback

class MyLoggingCallback(Callback):
    def on_step_end(self, engine, step, loss, **kwargs):
        if step % 100 == 0:
            print(f'Step {step}: loss={loss:.4f}')

    def on_epoch_end(self, engine, epoch, **kwargs):
        print(f'Epoch {epoch} complete')

engine = TrainerEngine(
    ...,
    callbacks=[MyLoggingCallback()],
)
```

---

## Common callback patterns

### Early stopping

```python
class EarlyStoppingCallback(Callback):
    def __init__(self, patience=3, metric='eval_loss'):
        self.patience = patience
        self.metric = metric
        self.best = None
        self.count = 0

    def on_epoch_end(self, engine, epoch, **kwargs):
        metrics = engine.trainer.evaluate()
        val = metrics.get(self.metric)
        if val is None:
            return
        if self.best is None or val < self.best:
            self.best = val
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                raise StopIteration('Early stopping triggered')
```

### Learning rate logging

```python
class LRLogCallback(Callback):
    def on_step_end(self, engine, step, loss, **kwargs):
        lr = engine.optimizer.param_groups[0]['lr']
        if wandb.run:
            wandb.log({'train/lr': lr}, step=step)
```

### Model checkpointing on improvement

```python
class BestModelCallback(Callback):
    def on_epoch_end(self, engine, epoch, **kwargs):
        metrics = engine.trainer.evaluate()
        if engine.trainer.update_best_metric(metrics):
            engine.trainer.save_checkpoint(is_best=True)
```

---

## Notes

- Callbacks receive `engine` as the first argument — giving access to
  `engine.trainer`, `engine.optimizer`, `engine.model`, etc.
- `TrainerEngine` catches `StopIteration` to support early stopping
- Multiple callbacks can be combined in a list
