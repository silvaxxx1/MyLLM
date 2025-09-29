# trainer/utils/training_utils.py
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def ensure_scalar_loss(loss_value) -> float:
    """
    Ensure loss is a scalar float value
    
    Args:
        loss_value: Loss value (Tensor, list, or scalar)
        
    Returns:
        float: Scalar loss value
        
    Raises:
        ValueError: If loss type is not supported
    """
    if isinstance(loss_value, torch.Tensor):
        return float(loss_value.item())
    elif isinstance(loss_value, list):
        return sum(float(l.item() if isinstance(l, torch.Tensor) else l) for l in loss_value)
    elif isinstance(loss_value, (int, float)):
        return float(loss_value)
    else:
        raise ValueError(f"Unsupported loss type: {type(loss_value)}")

def create_metrics_dict(step_results: Dict[str, Any], epoch: int, global_step: int, 
                       optimizer=None, latest_eval_loss: float = None) -> Dict[str, Any]:
    """
    Create standardized metrics dictionary for logging
    
    Args:
        step_results: Dictionary with step results
        epoch: Current epoch number
        global_step: Current global step number
        optimizer: Optimizer instance to get learning rate
        latest_eval_loss: Latest evaluation loss
        
    Returns:
        Dict[str, Any]: Metrics dictionary ready for logging
    """
    metrics = {
        "train/loss": step_results.get("loss", 0.0),
        "train/epoch": epoch,
        "train/step": global_step,
    }
    
    if optimizer:
        metrics["train/learning_rate"] = optimizer.param_groups[0]['lr']
    
    if latest_eval_loss is not None:
        metrics["eval/loss"] = latest_eval_loss
        
    return metrics

def handle_evaluation_metrics(eval_results: Dict[str, Any], global_step: int, 
                            optimizer=None, add_learning_rate: bool = True) -> Dict[str, Any]:
    """
    Process evaluation results into metrics format
    
    Args:
        eval_results: Dictionary with evaluation results
        global_step: Current global step number
        optimizer: Optimizer instance to get learning rate
        add_learning_rate: Whether to include learning rate in metrics
        
    Returns:
        Dict[str, Any]: Evaluation metrics dictionary
    """
    if not eval_results:
        return {}
    
    eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
    
    if add_learning_rate and optimizer:
        eval_metrics["eval/learning_rate"] = optimizer.param_groups[0]['lr']
        
    return eval_metrics