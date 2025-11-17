# trainer/sft_classifier_trainer.py (ENHANCED VERSION)
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
from collections import defaultdict

from .base_trainer import BaseTrainer
from .utils.progress_utils import create_progress_bar
from .utils.training_flow import TrainingFlow
from .utils.summary_utils import create_training_summary_table, print_training_completion
from .utils.model_utils import setup_model_compilation, load_pretrained_weights

import wandb

logger = logging.getLogger(__name__)

class AttentionPooling(nn.Module):
    """
    Learnable attention-based pooling mechanism
    Dynamically weights tokens based on their importance for classification
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, sequence_length, hidden_size)
            attention_mask: (batch_size, sequence_length)
        
        Returns:
            pooled_output: (batch_size, hidden_size)
        """
        # Compute attention weights
        attention_weights = self.attention(hidden_states)  # (batch_size, sequence_length, 1)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).float()
            attention_weights = attention_weights * attention_mask
            # Re-normalize attention weights
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        # Apply attention weights
        pooled_output = torch.sum(hidden_states * attention_weights, dim=1)
        return pooled_output

class MultiTaskClassificationHead(nn.Module):
    """
    Supports multiple classification tasks with shared encoder
    """
    
    def __init__(self, hidden_size: int, task_configs: Dict[str, int]):
        super().__init__()
        self.task_heads = nn.ModuleDict()
        
        for task_name, num_labels in task_configs.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_labels)
            )
    
    def forward(self, pooled_output: torch.Tensor, task_name: str) -> torch.Tensor:
        return self.task_heads[task_name](pooled_output)

class SFTClassifierTrainer(BaseTrainer):
    """
    Enhanced SFT Classifier Trainer with Advanced Features:
    - Multiple pooling strategies (including attention pooling)
    - Class imbalance handling
    - Multi-task learning support
    - Uncertainty quantification
    - Confidence calibration
    - Advanced monitoring
    - Gradient checkpointing
    """

    def __init__(self, config, model_config=None, model=None):
        # Initialize classification-specific attributes BEFORE calling super()
        self.num_labels = config.num_labels
        # Handle both enum and string pooling strategy
        if hasattr(config.pooling_strategy, 'value'):
            self.pooling_strategy = config.pooling_strategy.value
        else:
            self.pooling_strategy = config.pooling_strategy
        self.class_names = config.class_names
        
        # Initialize model-related attributes to None
        self.classifier = None
        self.attention_pooler = None
        self.multi_task_head = None
        self.all_predictions = []
        self.all_targets = []
        self.confidence_history = []
        
        # Enhanced attributes
        self.class_weights = getattr(config, 'class_weights', None)
        self.multi_task_config = getattr(config, 'multi_task_config', None)
        self.use_mc_dropout = getattr(config, 'use_mc_dropout', False)
        
        # FIX: Add training attribute
        self.training = False  # Default to evaluation mode
        
        # FIX: Add training_completed attribute
        self.training_completed = False
        
        # FIX: Call parent constructor FIRST before accessing any parent attributes
        super().__init__(config, model_config, model)
        
        # Use the shared training flow
        self.training_flow = TrainingFlow(self)

    def setup_model(self) -> torch.nn.Module:
        """Enhanced model setup with advanced features"""
        from myllm.api import LLM

        if self.model is not None:
            logger.info("Using externally provided model")
            self.model.to(self.device)
            self.setup_tokenizer()
            self._setup_classification_head()
            return self.model

        if self.model_config is None:
            raise ValueError("model_config must be provided to create a model")

        logger.info(f"Creating LLM for classification: {self.model_config.name} on {self.device}")
        self.llm = LLM(config=self.model_config, device=self.device)

        load_pretrained_weights(
            self.llm, 
            getattr(self.config, "pretrained_variant", None), 
            "gpt2"
        )

        self.model = self.llm.model
        self.setup_tokenizer()
        self.model.to(self.device)

        self._setup_classification_head()

        # Enable gradient checkpointing for memory efficiency
        if getattr(self.config, 'gradient_checkpointing', False):
            self._enable_gradient_checkpointing()

        self.model = setup_model_compilation(
            self.model, 
            self.config.use_compile,
            self.config
        )

        logger.info(f"Enhanced classifier setup with {self.num_labels} classes: {self.class_names}")
        return self.model

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory optimization"""
        if hasattr(self.model, 'transformer'):
            for block in self.model.transformer.values():
                if hasattr(block, 'gradient_checkpointing'):
                    block.gradient_checkpointing = True
            logger.info("✅ Gradient checkpointing enabled")
        else:
            logger.warning("❌ Gradient checkpointing not supported for this model")

    def _setup_classification_head(self):
        """Enhanced classification head with multiple architectures"""
        hidden_size = self.model_config.n_embd
        
        # Setup attention pooling if requested
        if self.pooling_strategy == 'attention':
            self.attention_pooler = AttentionPooling(hidden_size).to(self.device)
            logger.info("✅ Attention pooling initialized")

        # Setup multi-task learning if configured
        if self.multi_task_config:
            self.multi_task_head = MultiTaskClassificationHead(
                hidden_size, self.multi_task_config
            ).to(self.device)
            logger.info(f"✅ Multi-task head initialized for tasks: {list(self.multi_task_config.keys())}")
            return

        # Standard single-task classifier
        classifier_layers = []
        dropout_rate = self.config.classifier_dropout
        
        if self.config.classifier_hidden_size:
            classifier_layers.extend([
                nn.Linear(hidden_size, self.config.classifier_hidden_size),
                nn.GELU(),  # Better than ReLU for deep networks
                nn.Dropout(dropout_rate),
                nn.Linear(self.config.classifier_hidden_size, self.num_labels)
            ])
        else:
            classifier_layers.extend([
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, self.num_labels)
            ])
        
        self.classifier = nn.Sequential(*classifier_layers).to(self.device)
        logger.info(f"Classification head created: {hidden_size} -> {self.num_labels} labels")

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Enhanced hidden states extraction with gradient checkpointing support
        """
        # Use gradient checkpointing for memory efficiency
        if getattr(self.config, 'gradient_checkpointing', False) and self.training:
            return self._get_hidden_states_with_checkpointing(input_ids)
        
        if hasattr(self.model, 'forward_hidden_states'):
            return self.model.forward_hidden_states(input_ids)
        else:
            # Fallback to manual extraction
            device = input_ids.device
            B, T = input_ids.size()
            
            # Token embeddings
            token_embeddings = self.model.wte(input_ids)
            
            # Position embeddings (if using learned embeddings)
            if hasattr(self.model, 'wpe'):
                pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
                position_embeddings = self.model.wpe(pos)
                x = token_embeddings + position_embeddings
            else:
                x = token_embeddings
            
            # Forward through transformer blocks
            for block in self.model.transformer.values():
                x = block(x)
            
            # Apply final layer norm
            x = self.model.ln_f(x)
            
            return x

    def _get_hidden_states_with_checkpointing(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Hidden states extraction with gradient checkpointing"""
        if hasattr(self.model, 'forward_hidden_states'):
            return torch.utils.checkpoint.checkpoint(
                self.model.forward_hidden_states, 
                input_ids,
                use_reentrant=False
            )
        else:
            # Manual implementation with checkpointing
            device = input_ids.device
            B, T = input_ids.size()
            
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(inputs[0])
                return custom_forward
            
            # Token embeddings (no checkpointing - lightweight)
            token_embeddings = self.model.wte(input_ids)
            
            # Position embeddings
            if hasattr(self.model, 'wpe'):
                pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
                position_embeddings = self.model.wpe(pos)
                x = token_embeddings + position_embeddings
            else:
                x = token_embeddings
            
            # Forward through transformer blocks with checkpointing
            for block in self.model.transformer.values():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), 
                    x,
                    use_reentrant=False
                )
            
            # Final layer norm
            x = self.model.ln_f(x)
            
            return x

    def setup_optimizer(self):
        """Enhanced optimizer with differential learning rates and multi-task support"""
        if self.model is None:
            raise ValueError("Model must be setup before optimizer")
        
        # Use config learning rates directly
        base_lr = self.config.base_model_learning_rate
        classifier_lr = self.config.classifier_learning_rate
        
        # Collect all parameters
        base_params = []
        classifier_params = []
        attention_pooler_params = []
        multi_task_params = []
        
        for name, param in self.model.named_parameters():
            base_params.append(param)
        
        if self.classifier is not None:
            for param in self.classifier.parameters():
                classifier_params.append(param)
                
        if self.attention_pooler is not None:
            for param in self.attention_pooler.parameters():
                attention_pooler_params.append(param)
                
        if self.multi_task_head is not None:
            for param in self.multi_task_head.parameters():
                multi_task_params.append(param)
        
        # Create optimizer groups
        optimizer_groups = [
            {"params": base_params, "lr": base_lr, "name": "base_model"},
        ]
        
        if classifier_params:
            optimizer_groups.append({"params": classifier_params, "lr": classifier_lr, "name": "classifier"})
        if attention_pooler_params:
            optimizer_groups.append({"params": attention_pooler_params, "lr": classifier_lr, "name": "attention_pooler"})
        if multi_task_params:
            optimizer_groups.append({"params": multi_task_params, "lr": classifier_lr, "name": "multi_task"})
        
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            weight_decay=self.model_config.weight_decay,
            betas=(self.model_config.beta1, self.model_config.beta2)
        )
        
        # Use parent's scheduler setup logic
        if (self.config.scheduler_type.value == "linear" and 
            hasattr(self.config, "warmup_steps")):
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        else:
            self.scheduler = None
        
        logger.info(f"Enhanced optimizer with differential learning rates: base={base_lr:.2e}, classifier={classifier_lr:.2e}")

    def _prepare_batch(self, batch):
        """Enhanced batch preparation for multi-task learning"""
        prepared_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Handle multi-task labels
        if self.multi_task_head and 'task_labels' in batch:
            prepared_batch['task_labels'] = {
                task: labels.to(self.device) for task, labels in batch['task_labels'].items()
            }
            
        return prepared_batch

    def _get_labels(self, batch, task_name: str = 'main'):
        """Enhanced label handling for multi-task scenarios"""
        if self.multi_task_head:
            return batch.get("task_labels", {}).get(task_name, batch.get("labels"))
        return batch.get("labels")

    def _pool_hidden_states(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced pooling with attention-based strategy
        """
        if self.pooling_strategy == 'attention' and self.attention_pooler is not None:
            return self.attention_pooler(hidden_states, attention_mask)
            
        elif self.pooling_strategy == 'last':
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.shape[0]
                pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
            else:
                pooled = hidden_states[:, -1]
                
        elif self.pooling_strategy == 'mean':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = hidden_states.mean(dim=1)
                
        elif self.pooling_strategy == 'max':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                hidden_states = hidden_states.clone()
                hidden_states[mask_expanded == 0] = -1e9
            pooled = torch.max(hidden_states, dim=1)[0]
            
        elif self.pooling_strategy == 'cls':
            pooled = hidden_states[:, 0]
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return pooled

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, task_name: str = 'main') -> torch.Tensor:
        """Enhanced loss computation with class balancing and multi-task support"""
        # Handle class imbalance
        if self.class_weights is not None and task_name in self.class_weights:
            weights = torch.tensor(self.class_weights[task_name]).to(self.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        elif hasattr(self.config, 'label_smoothing') and self.config.label_smoothing > 0:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        else:
            loss_fn = nn.CrossEntropyLoss()
            
        return loss_fn(logits, labels)

    def forward(self, batch, task_name: str = 'main') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced forward pass with multi-task support
        Returns: (logits, hidden_states)
        """
        # Get hidden states
        hidden_states = self._get_hidden_states(batch["input_ids"])
        
        # Pool hidden states
        pooled_output = self._pool_hidden_states(
            hidden_states, 
            batch.get("attention_mask")
        )
        
        # Get logits
        if self.multi_task_head:
            logits = self.multi_task_head(pooled_output, task_name)
        else:
            logits = self.classifier(pooled_output)
            
        return logits, hidden_states

    def training_step(self, batch) -> Dict[str, Any]:
        """Enhanced training step with multi-task support and confidence tracking"""
        self.model.train()
        if self.classifier:
            self.classifier.train()
        if self.multi_task_head:
            self.multi_task_head.train()
        if self.attention_pooler:
            self.attention_pooler.train()
        
        batch = self._prepare_batch(batch)
        
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            # Handle multi-task training
            if self.multi_task_head:
                total_loss = 0
                all_predictions = []
                all_targets = []
                
                for task_name in self.multi_task_config.keys():
                    logits, _ = self.forward(batch, task_name)
                    labels = self._get_labels(batch, task_name)
                    
                    if labels is not None:
                        task_loss = self.compute_loss(logits, labels, task_name)
                        total_loss += task_loss
                        
                        predictions = torch.argmax(logits, dim=-1)
                        all_predictions.extend(predictions.cpu().numpy())
                        all_targets.extend(labels.cpu().numpy())
                
                loss = total_loss / len(self.multi_task_config)
                accuracy = (torch.tensor(all_predictions) == torch.tensor(all_targets)).float().mean().item()
                
            else:
                # Single-task training
                logits, _ = self.forward(batch)
                labels = self._get_labels(batch)
                
                loss = self.compute_loss(logits, labels)
                loss = loss / max(1, self.config.gradient_accumulation_steps)
                
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == labels).float().mean().item()
        
        self.scaler.scale(loss).backward()
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                # Collect all parameters for gradient clipping
                all_parameters = list(self.model.parameters())
                if self.classifier:
                    all_parameters.extend(list(self.classifier.parameters()))
                if self.attention_pooler:
                    all_parameters.extend(list(self.attention_pooler.parameters()))
                if self.multi_task_head:
                    all_parameters.extend(list(self.multi_task_head.parameters()))
                    
                torch.nn.utils.clip_grad_norm_(
                    all_parameters, 
                    self.config.max_grad_norm
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
        
        # Track confidence metrics
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=-1)
            max_probs, _ = torch.max(probabilities, dim=-1)
            avg_confidence = max_probs.mean().item()
            self.confidence_history.append(avg_confidence)
        
        return {
            "loss": float(loss.item() * max(1, self.config.gradient_accumulation_steps)),
            "accuracy": accuracy,
            "confidence": avg_confidence,
            "predictions": predictions.cpu().numpy() if not self.multi_task_head else all_predictions,
            "targets": labels.cpu().numpy() if not self.multi_task_head else all_targets
        }

    def evaluation_step(self, batch) -> Dict[str, float]:
        """Enhanced evaluation step with uncertainty estimation"""
        self.model.eval()
        if self.classifier:
            self.classifier.eval()
        if self.multi_task_head:
            self.multi_task_head.eval()
        if self.attention_pooler:
            self.attention_pooler.eval()
        
        batch = self._prepare_batch(batch)
        
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            if self.multi_task_head:
                total_loss = 0
                total_correct = 0
                total_samples = 0
                all_predictions = []
                all_targets = []
                
                for task_name in self.multi_task_config.keys():
                    logits, _ = self.forward(batch, task_name)
                    labels = self._get_labels(batch, task_name)
                    
                    if labels is not None:
                        task_loss = self.compute_loss(logits, labels, task_name)
                        total_loss += task_loss.item() * labels.size(0)
                        
                        predictions = torch.argmax(logits, dim=-1)
                        correct = (predictions == labels).sum().item()
                        total_correct += correct
                        total_samples += labels.size(0)
                        
                        all_predictions.extend(predictions.cpu().numpy())
                        all_targets.extend(labels.cpu().numpy())
                
                return {
                    "loss": total_loss,
                    "correct": total_correct,
                    "total": total_samples,
                    "predictions": all_predictions,
                    "targets": all_targets
                }
            else:
                # Single-task evaluation
                logits, _ = self.forward(batch)
                labels = self._get_labels(batch)
                
                loss = self.compute_loss(logits, labels)
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == labels).sum().item()
                
                return {
                    "loss": loss.item() * labels.size(0),
                    "correct": correct,
                    "total": labels.size(0),
                    "predictions": predictions.cpu().numpy(),
                    "targets": labels.cpu().numpy()
                }

    def mc_dropout_predict(self, batch, num_samples: int = 30, task_name: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout for uncertainty estimation
        Returns: (mean_predictions, uncertainty_scores)
        """
        if not self.use_mc_dropout:
            raise ValueError("MC dropout not enabled in config")
            
        self.model.eval()
        
        # Enable dropout for uncertainty estimation
        if self.classifier:
            self.classifier.train()  # Keep dropout active
        if self.multi_task_head:
            self.multi_task_head.train()  # Keep dropout active
        if self.attention_pooler:
            self.attention_pooler.train()  # Keep dropout active
        
        all_predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # FIX: Handle multi-task
                if self.multi_task_head:
                    if task_name is None:
                        task_name = list(self.multi_task_config.keys())[0]
                    logits, _ = self.forward(batch, task_name)
                else:
                    logits, _ = self.forward(batch)
                    
                probabilities = torch.softmax(logits, dim=-1)
                all_predictions.append(probabilities.unsqueeze(0))
        
        # Stack all samples
        all_predictions = torch.cat(all_predictions, dim=0)  # (num_samples, batch_size, num_classes)
        
        # Compute mean and uncertainty
        mean_predictions = all_predictions.mean(dim=0)
        uncertainty = all_predictions.std(dim=0).mean(dim=-1)  # Mean std across classes
        
        return mean_predictions, uncertainty

    def predict_with_confidence(self, batch, task_name: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced prediction with confidence scores
        Returns: (predictions, confidence_scores)
        """
        self.model.eval()
        if self.classifier:
            self.classifier.eval()
        if self.multi_task_head:
            self.multi_task_head.eval()
        if self.attention_pooler:
            self.attention_pooler.eval()
        
        with torch.no_grad():
            # FIX: Handle multi-task vs single-task scenarios
            if self.multi_task_head:
                # For multi-task, use the first task or specified task
                if task_name is None:
                    # Use the first task in the config
                    task_name = list(self.multi_task_config.keys())[0]
                logits, _ = self.forward(batch, task_name)
            else:
                # Single-task
                logits, _ = self.forward(batch)
                
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predictions = torch.max(probabilities, dim=-1)
            
        return predictions, confidence

    def evaluate(self) -> Dict[str, float]:
        """Enhanced evaluation with uncertainty and confidence metrics"""
        if self.eval_dataloader is None:
            return {}
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        all_confidence_scores = []
        
        for batch in self.eval_dataloader:
            step_results = self.evaluation_step(batch)
            total_loss += step_results["loss"]
            total_correct += step_results["correct"]
            total_samples += step_results["total"]
            all_predictions.extend(step_results["predictions"])
            all_targets.extend(step_results["targets"])
            
            # Get confidence scores
            if not self.multi_task_head:  # Confidence for single task only
                _, confidence = self.predict_with_confidence(batch)
                all_confidence_scores.extend(confidence.cpu().numpy())
        
        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        
        # Enhanced classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        self.latest_eval_loss = avg_loss
        self.all_predictions = all_predictions
        self.all_targets = all_targets
        
        # Enhanced logging with confidence metrics
        self._log_classification_metrics(all_targets, all_predictions, all_confidence_scores)
        
        results = {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1
        }
        
        # Add confidence metrics
        if all_confidence_scores:
            results.update({
                "eval_confidence_mean": np.mean(all_confidence_scores),
                "eval_confidence_std": np.std(all_confidence_scores),
                "eval_low_confidence_ratio": np.mean(np.array(all_confidence_scores) < 0.7)
            })
        
        return results

    def _log_classification_metrics(self, targets: List[int], predictions: List[int], confidence_scores: List[float] = None):
        """Enhanced classification metrics logging with confidence analysis"""
        if not targets or not predictions:
            return
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Per-class metrics
        class_report = classification_report(
            targets, predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Log to WandB
        if wandb.run is not None:
            try:
                # Confusion matrix
                wandb.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=targets,
                        preds=predictions,
                        class_names=self.class_names
                    )
                }, step=self.global_step)
                
                # Enhanced metrics table
                metrics_table = wandb.Table(columns=["Metric", "Value"])
                metrics_table.add_data("Accuracy", class_report['accuracy'])
                metrics_table.add_data("Weighted F1", class_report['weighted avg']['f1-score'])
                metrics_table.add_data("Macro F1", class_report['macro avg']['f1-score'])
                
                if confidence_scores:
                    metrics_table.add_data("Avg Confidence", np.mean(confidence_scores))
                    metrics_table.add_data("Low Confidence Ratio", np.mean(np.array(confidence_scores) < 0.7))
                
                wandb.log({"classification_metrics": metrics_table}, step=self.global_step)
                
                # Confidence distribution
                if confidence_scores:
                    wandb.log({
                        "confidence_distribution": wandb.Histogram(confidence_scores)
                    }, step=self.global_step)
                
            except Exception as e:
                logger.warning(f"WandB logging failed: {e}")
        
        # Enhanced console logging
        logger.info(f"Classification Metrics - Accuracy: {class_report['accuracy']:.4f}, "
                   f"F1: {class_report['weighted avg']['f1-score']:.4f}")
        
        if confidence_scores:
            logger.info(f"Confidence Stats - Mean: {np.mean(confidence_scores):.4f}, "
                       f"Low Confidence (<0.7): {np.mean(np.array(confidence_scores) < 0.7):.4f}")

    def _log_training_insights(self, predictions: np.ndarray, targets: np.ndarray, logits: torch.Tensor):
        """
        Enhanced training insights with confidence and difficulty analysis
        """
        if wandb.run is None:
            return
            
        try:
            probabilities = torch.softmax(logits, dim=-1)
            max_probs, _ = torch.max(probabilities, dim=-1)
            
            # Track prediction confidence distribution
            wandb.log({
                "training/confidence_mean": max_probs.mean().item(),
                "training/confidence_std": max_probs.std().item(),
                "training/hard_examples_ratio": (max_probs < 0.7).float().mean().item(),
                "training/very_confident_ratio": (max_probs > 0.9).float().mean().item()
            }, step=self.global_step)
            
            # Track class-wise performance
            for class_idx, class_name in enumerate(self.class_names):
                class_mask = targets == class_idx
                if class_mask.any():
                    class_accuracy = (predictions[class_mask] == targets[class_mask]).mean()
                    class_confidence = max_probs[class_mask].mean().item() if len(max_probs) > 0 else 0
                    wandb.log({
                        f"class_{class_name}/accuracy": class_accuracy,
                        f"class_{class_name}/confidence": class_confidence
                    }, step=self.global_step)
                    
        except Exception as e:
            logger.warning(f"Training insights logging failed: {e}")

    def setup_data(self, train_dataloader=None, eval_dataloader=None):
        """Setup training and validation data - REQUIRED by BaseTrainer"""
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        if eval_dataloader is not None:
            self.eval_dataloader = eval_dataloader
        return self.train_dataloader, self.eval_dataloader

    def save_checkpoint(self, checkpoint_dir: Optional[str] = None, is_best: bool = False):
        """Enhanced checkpoint saving with all components"""
        # First call parent to save base model and training state
        checkpoint_dir = super().save_checkpoint(checkpoint_dir, is_best)
        
        # Save all additional components
        components = {
            "classifier": self.classifier,
            "attention_pooler": self.attention_pooler,
            "multi_task_head": self.multi_task_head
        }
        
        for name, component in components.items():
            if component is not None:
                component_path = os.path.join(checkpoint_dir, f"{name}.bin")
                torch.save(component.state_dict(), component_path)
                logger.info(f"✅ {name} saved to {component_path}")
        
        # Save confidence history
        confidence_path = os.path.join(checkpoint_dir, "confidence_history.npy")
        np.save(confidence_path, np.array(self.confidence_history))
        
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir: str):
        """Enhanced checkpoint loading with all components"""
        # First call parent to load base model and training state
        super().load_checkpoint(checkpoint_dir)
        
        # Load all additional components
        components = {
            "classifier": self.classifier,
            "attention_pooler": self.attention_pooler,
            "multi_task_head": self.multi_task_head
        }
        
        for name, component in components.items():
            if component is not None:
                component_path = os.path.join(checkpoint_dir, f"{name}.bin")
                if os.path.exists(component_path):
                    component.load_state_dict(torch.load(component_path, map_location=self.device))
                    logger.info(f"✅ {name} loaded from {component_path}")
        
        # Load confidence history
        confidence_path = os.path.join(checkpoint_dir, "confidence_history.npy")
        if os.path.exists(confidence_path):
            self.confidence_history = np.load(confidence_path).tolist()
        
        logger.info("✅ Enhanced checkpoint loading completed")

    def train(self):
        """Enhanced training loop with advanced monitoring"""
        if self.train_dataloader is None:
            logger.warning("No training data available")
            return

        self.setup_wandb()
        
        from rich.console import Console
        console = Console()
        
        # Enhanced training header
        console.rule(f"[bold green]🚀 Starting Enhanced SFT Classification Training")
        console.print(f"[bold blue]📊 Classes: {self.class_names}[/bold blue]")
        console.print(f"[bold cyan]🎯 Pooling: {self.pooling_strategy}[/bold cyan]")
        if self.multi_task_head:
            console.print(f"[bold magenta]🔄 Multi-task: {list(self.multi_task_config.keys())}[/bold magenta]")
        if self.class_weights:
            console.print(f"[bold yellow]⚖️ Class weights: Enabled[/bold yellow]")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Enhanced progress bar
            progress = create_progress_bar(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            task = progress.add_task(
                f"Epoch {epoch + 1}/{self.config.num_epochs}",
                total=len(self.train_dataloader)
            )

            with progress:
                epoch_loss, num_steps = self.training_flow.run_epoch(epoch, progress, task)

            self.training_flow.handle_end_of_epoch(epoch)

        # Enhanced summary with all metrics
        summary_table = create_training_summary_table(
            "🚀 Enhanced SFT Classification Training Summary", 
            self.config, 
            self.latest_eval_loss, 
            self.best_checkpoint_path, 
            self.wandb_url
        )
        
        # Add enhanced metrics
        if hasattr(self, 'all_predictions') and self.all_predictions:
            accuracy = accuracy_score(self.all_targets, self.all_predictions)
            summary_table.add_row("Final Accuracy", f"{accuracy:.4f}")
            
        if self.confidence_history:
            avg_confidence = np.mean(self.confidence_history[-100:])  # Last 100 steps
            summary_table.add_row("Avg Confidence", f"{avg_confidence:.4f}")
        
        console.print(summary_table)
        print_training_completion("🎉 Enhanced SFT Classification Training Completed!")
        
        # FIX: Set training_completed to True
        self.training_completed = True
        
        if wandb.run is not None:
            wandb.finish()

    def debug_model_outputs(self, input_ids: torch.Tensor):
        """
        Enhanced debug method with comprehensive analysis
        """
        print(f"\n🔍 ENHANCED DEBUG MODEL OUTPUTS:")
        print(f"Input shape: {input_ids.shape}")
        
        # Test 1: Direct model call
        print(f"\n1. Direct model call:")
        try:
            direct_output = self.model(input_ids)
            print(f"   ✅ Direct output shape: {direct_output.shape}")
            print(f"   ✅ Direct output type: {type(direct_output)}")
        except Exception as e:
            print(f"   ❌ Direct call failed: {e}")
        
        # Test 2: Enhanced architecture analysis
        print(f"\n2. Enhanced model architecture:")
        print(f"   Model type: {type(self.model)}")
        print(f"   Pooling strategy: {self.pooling_strategy}")
        print(f"   Multi-task enabled: {self.multi_task_head is not None}")
        print(f"   Attention pooling: {self.attention_pooler is not None}")
        print(f"   Gradient checkpointing: {getattr(self.config, 'gradient_checkpointing', False)}")
        
        # Test 3: Comprehensive forward pass test
        print(f"\n3. Comprehensive forward pass:")
        try:
            # Test hidden states extraction
            hidden_states = self._get_hidden_states(input_ids)
            print(f"   ✅ Hidden states shape: {hidden_states.shape}")
            
            # Test pooling
            pooled = self._pool_hidden_states(hidden_states)
            print(f"   ✅ Pooled shape: {pooled.shape}")
            
            # Test classification
            if self.multi_task_head:
                for task_name in self.multi_task_config.keys():
                    logits = self.multi_task_head(pooled, task_name)
                    print(f"   ✅ {task_name} logits shape: {logits.shape}")
            else:
                logits = self.classifier(pooled)
                print(f"   ✅ Classifier logits shape: {logits.shape}")
                
            # Test confidence - FIXED for multi-task
            try:
                if self.multi_task_head:
                    # Use sentiment task for testing
                    predictions, confidence = self.predict_with_confidence(
                        {"input_ids": input_ids}, task_name="sentiment"
                    )
                else:
                    predictions, confidence = self.predict_with_confidence(
                        {"input_ids": input_ids}
                    )
                print(f"   ✅ Predictions shape: {predictions.shape}")
                print(f"   ✅ Confidence shape: {confidence.shape}")
                print(f"   🎉 ENHANCED SUCCESS: All components working!")
                
            except Exception as e:
                print(f"   ❌ Confidence test failed: {e}")
            
        except Exception as e:
            print(f"   ❌ Comprehensive test failed: {e}")
            import traceback
            traceback.print_exc()

    def update_best_metric(self, eval_results, key: str = None):
        """Enhanced best metric tracking with multi-task support"""
        if key is None:
            key = self.config.metric_for_best_model
            
        # For multi-task, you might want to track a composite metric
        if self.multi_task_head and key == "eval_loss":
            # Optionally create a composite metric for multi-task
            pass
            
        return super().update_best_metric(eval_results, key)

# Configuration example for the enhanced trainer
"""
Example config for enhanced features:

class EnhancedTrainerConfig:
    # Existing config...
    num_labels = 3
    class_names = ["negative", "neutral", "positive"]
    pooling_strategy = "attention"  # Options: 'last', 'mean', 'max', 'cls', 'attention'
    
    # Enhanced features
    class_weights = [0.8, 1.0, 1.2]  # Handle class imbalance
    use_mc_dropout = True  # Enable uncertainty estimation
    gradient_checkpointing = True  # Memory optimization
    
    # Multi-task learning
    multi_task_config = {
        "sentiment": 3,      # 3 classes for sentiment
        "topic": 5,          # 5 classes for topic
        "urgency": 2         # 2 classes for urgency
    }
    
    # Advanced optimizer
    base_model_learning_rate = 1e-5
    classifier_learning_rate = 1e-4
"""