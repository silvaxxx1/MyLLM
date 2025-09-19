# example_pipeline.py

# ---------------------------------------------------------------------
# Import the Single GPU accelerator class - used for handling single-GPU training
# from engine.accelerators.single_gpu import SingleGPUAccelerator

# Import the optimizer manager - handles creation and management of optimizers (e.g., AdamW, SGD)
# from engine.optimizer import OptimizerManager

# Import the learning rate scheduler manager - manages learning rate schedules
# from engine.lr_scheduler import SchedulerManager

# Import the trainer engine - orchestrates the whole training loop
# from engine.trainer_engine import TrainerEngine

# Import checkpoint manager - handles saving and loading model checkpoints
# from engine.checkpoint_manager import CheckpointManager

# Import callback system - e.g., printing logs or metrics during training
# from engine.callbacks import PrintCallback
# ---------------------------------------------------------------------

# NOTE: The `YourTrainer` class must be implemented by the user.
# It should define the following:
#   - `model`: The model to train
#   - `train_dataloader()`: DataLoader that provides training batches
#   - `train_step(batch)`: A single forward & backward step
#   - `batch_to_device(batch, device)`: Moves data to the correct device
# ---------------------------------------------------------------------

# trainer = YourTrainer(...)  # <- Replace with your custom trainer class

# Configuration dictionary for the training pipeline
# config = {
#     "num_epochs": 3,            # Number of epochs to train
#     "gradient_clip": 1.0,        # Max gradient norm for clipping
#     "save_every_epoch": True,    # Save checkpoint after every epoch
#     "optimizer": {               # Optimizer configuration
#         "name": "adamw",         # Optimizer type (e.g., adamw, sgd)
#         "lr": 1e-4               # Learning rate
#     }
# }

# Initialize the accelerator for single GPU training
# acc = SingleGPUAccelerator(config)

# Initialize the optimizer manager with the model and config
# opt_mgr = OptimizerManager(trainer.model, config)

# Initialize the scheduler manager
# (Currently passing `None` as the optimizer, to be supplied after creation)
# sched_mgr = SchedulerManager(None, {})

# Initialize the checkpoint manager to save and load model checkpoints
# ckpt_mgr = CheckpointManager(trainer.model)

# Build the training engine with all components
# engine = TrainerEngine(
#     trainer,                      # User-defined trainer instance
#     acc,                          # Accelerator for handling devices
#     opt_mgr,                       # Optimizer manager
#     scheduler_manager=sched_mgr,   # LR scheduler manager
#     checkpoint_manager=ckpt_mgr,   # Checkpoint manager
#     callbacks=[PrintCallback()],   # List of callbacks for logging, etc.
#     config=config                  # Configuration dictionary
# )

# Setup the training engine
# engine.setup()

# Start the training process
# engine.train()
