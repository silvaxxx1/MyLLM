# trainer/utils/summary_utils.py
from rich.table import Table
from rich.console import Console

console = Console()

def create_training_summary_table(title: str, config, latest_eval_loss: float = None, 
                                best_checkpoint_path: str = None, wandb_url: str = "N/A") -> Table:
    """
    Create a standardized training summary table
    
    Args:
        title: Table title
        config: Trainer configuration object
        latest_eval_loss: Latest evaluation loss value
        best_checkpoint_path: Path to best checkpoint
        wandb_url: WandB run URL
        
    Returns:
        Table: Configured Rich Table instance
    """
    table = Table(title=title, show_lines=True)
    table.add_column("Epochs", justify="center")
    table.add_column("Final Eval Loss", justify="center")
    table.add_column("Best Checkpoint", justify="center")
    table.add_column("W&B Run URL", justify="center")

    table.add_row(
        str(getattr(config, 'num_epochs', 'N/A')),
        f"{latest_eval_loss:.4f}" if latest_eval_loss is not None else "N/A",
        best_checkpoint_path if best_checkpoint_path else "N/A",
        wandb_url
    )
    
    return table

def print_training_completion(message: str):
    """
    Print standardized training completion message
    
    Args:
        message: Completion message to display
    """
    console.rule(f"[bold green]{message}")