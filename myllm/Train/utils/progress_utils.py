# trainer/utils/progress_utils.py
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from typing import Dict, Any

console = Console()

def create_progress_bar(description: str) -> Progress:
    """
    Create a standardized progress bar for training
    
    Args:
        description: Description for the progress bar
        
    Returns:
        Progress: Configured Rich Progress instance
    """
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[yellow]LR: {task.fields[lr]:.2e}"),
        TextColumn("[red]Train Loss: {task.fields[train_loss]:.4f}"),
        TextColumn("[cyan]Avg Loss: {task.fields[avg_loss]:.4f}"),
        TextColumn("[magenta]Eval Loss: {task.fields[eval_loss]:.4f}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

def update_progress_bar(progress, task, step_results: Dict[str, Any], epoch_loss: float, num_steps: int, 
                       optimizer=None, latest_eval_loss: float = None):
    """
    Update progress bar with current training metrics
    
    Args:
        progress: Rich Progress instance
        task: Progress task ID
        step_results: Dictionary with step results (must contain 'loss')
        epoch_loss: Cumulative loss for the epoch
        num_steps: Number of steps completed in epoch
        optimizer: Optimizer instance to get learning rate
        latest_eval_loss: Latest evaluation loss for display
    """
    avg_loss = epoch_loss / max(1, num_steps)
    lr = optimizer.param_groups[0]['lr'] if optimizer else 0.0
    eval_loss = latest_eval_loss or 0.0

    progress.update(
        task,
        advance=1,
        train_loss=step_results.get('loss', 0.0),
        avg_loss=avg_loss,
        eval_loss=eval_loss,
        lr=lr
    )