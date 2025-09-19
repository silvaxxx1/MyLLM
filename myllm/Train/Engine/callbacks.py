# engine/callbacks.py
class Callback:
    """Base callback with common hook names."""
    def on_train_start(self, engine, **kwargs): pass
    def on_train_end(self, engine, **kwargs): pass
    def on_epoch_start(self, engine, **kwargs): pass
    def on_epoch_end(self, engine, **kwargs): pass
    def on_step_end(self, engine, **kwargs): pass
    def on_validation_end(self, engine, **kwargs): pass

# Example callback (simple logger)
class PrintCallback(Callback):
    def on_epoch_end(self, engine, epoch, metrics=None, **kwargs):
        print(f"[Epoch {epoch}] metrics: {metrics}")
