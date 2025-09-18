# engine/utils.py
class EngineUtils:
    """Helper functions: gradient clipping, mixed-precision, batch utilities"""

    @staticmethod
    def clip_gradients(model, max_norm):
        """Placeholder for gradient clipping"""
        pass

    @staticmethod
    def batch_to_device(batch, device):
        """Move batch to device"""
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
