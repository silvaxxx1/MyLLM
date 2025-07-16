import os
import torch

def save_checkpoint(trainer, step):
    ckpt_name = f"checkpoint-{step}.pt"
    save_path = os.path.join(trainer.config.output_dir, ckpt_name)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    torch.save({
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'step': trainer.global_step
    }, save_path)
    print(f"✅ Saved checkpoint: {save_path}")

    if trainer.config.save_total_limit is not None:
        cleanup_old_checkpoints(trainer)

def load_checkpoint(trainer, path):
    checkpoint = torch.load(path, map_location=trainer.accelerator.device)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    trainer.global_step = checkpoint['step']
    print(f"✅ Loaded checkpoint from {path}")

def cleanup_old_checkpoints(trainer):
    files = sorted(
        [f for f in os.listdir(trainer.config.output_dir) if f.startswith("checkpoint-") and f.endswith(".pt")],
        key=lambda x: os.path.getmtime(os.path.join(trainer.config.output_dir, x))
    )
    while len(files) > trainer.config.save_total_limit:
        oldest = files.pop(0)
        os.remove(os.path.join(trainer.config.output_dir, oldest))
        print(f"🗑 Removed old checkpoint: {oldest}")
