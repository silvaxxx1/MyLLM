# trainer/sft_classifier_trainer.py (FINAL CLEAN VERSION)
import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

import os

from .base_trainer import BaseTrainer
from .utils.progress_utils import create_progress_bar
from .utils.training_flow import TrainingFlow
from .utils.summary_utils import create_training_summary_table, print_training_completion
from .utils.model_utils import setup_model_compilation, load_pretrained_weights

import wandb

logger = logging.getLogger(__name__)

class SFTClassifierTrainer(BaseTrainer):
    """
    Optimized SFT Classifier Trainer - Properly extends BaseTrainer
    Only implements classification-specific logic
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
        self.classifier = None
        self.all_predictions = []
        self.all_targets = []
        
        # Now call parent constructor
        super().__init__(config, model_config, model)
        
        # Use the shared training flow
        self.training_flow = TrainingFlow(self)

    def setup_model(self) -> torch.nn.Module:
        """Setup model with classification head - extends parent method"""
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

        self.model = setup_model_compilation(
            self.model, 
            self.config.use_compile,
            self.config
        )

        logger.info(f"Classifier setup with {self.num_labels} classes: {self.class_names}")
        return self.model

    def _setup_classification_head(self):
        """Setup classification head - classification-specific"""
        hidden_size = self.model_config.n_embd
        
        classifier_layers = []
        dropout_rate = self.config.classifier_dropout
        
        if self.config.classifier_hidden_size:
            classifier_layers.extend([
                nn.Linear(hidden_size, self.config.classifier_hidden_size),
                nn.GELU(),
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
        Use the model's forward_hidden_states method if available
        Otherwise fall back to manual extraction
        """
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

    def setup_optimizer(self):
        """Enhanced optimizer with differential learning rates - extends parent"""
        if self.model is None or self.classifier is None:
            raise ValueError("Model and classifier must be setup before optimizer")
        
        # Use config learning rates directly
        base_lr = self.config.base_model_learning_rate
        classifier_lr = self.config.classifier_learning_rate
        
        # Separate parameter groups
        base_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            base_params.append(param)
        
        for param in self.classifier.parameters():
            classifier_params.append(param)
        
        optimizer_groups = [
            {"params": base_params, "lr": base_lr, "name": "base_model"},
            {"params": classifier_params, "lr": classifier_lr, "name": "classifier"}
        ]
        
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
        """Prepare batch for training/evaluation - REQUIRED by BaseTrainer"""
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _get_labels(self, batch):
        """Get classification labels from batch - REQUIRED by BaseTrainer"""
        return batch.get("labels")

    def _pool_hidden_states(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool hidden states to get sequence representation - Classification-specific
        """
        if self.pooling_strategy == 'last':
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

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Classification loss computation - Overrides parent method"""
        if hasattr(self.config, 'label_smoothing') and self.config.label_smoothing > 0:
            return nn.functional.cross_entropy(
                logits, 
                labels, 
                label_smoothing=self.config.label_smoothing
            )
        else:
            return nn.functional.cross_entropy(logits, labels)

    def training_step(self, batch) -> Dict[str, Any]:
        """Enhanced training step with classification metrics - Overrides parent"""
        self.model.train()
        self.classifier.train()
        
        batch = self._prepare_batch(batch)
        
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            # Get hidden states instead of logits - FIXED
            hidden_states = self._get_hidden_states(batch["input_ids"])
            pooled_output = self._pool_hidden_states(
                hidden_states, 
                batch.get("attention_mask")
            )
            logits = self.classifier(pooled_output)
            labels = self._get_labels(batch)
            
            loss = self.compute_loss(logits, labels)
            loss = loss / max(1, self.config.gradient_accumulation_steps)
        
        self.scaler.scale(loss).backward()
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.classifier.parameters()), 
                    self.config.max_grad_norm
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
        
        # Classification-specific metrics
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean().item()
        
        return {
            "loss": float(loss.item()),
            "accuracy": accuracy,
            "predictions": predictions.cpu().numpy(),
            "targets": labels.cpu().numpy()
        }

    def evaluation_step(self, batch) -> Dict[str, float]:
        """Enhanced evaluation step with prediction collection - Overrides parent"""
        self.model.eval()
        self.classifier.eval()
        
        batch = self._prepare_batch(batch)
        
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            # Get hidden states instead of logits - FIXED
            hidden_states = self._get_hidden_states(batch["input_ids"])
            pooled_output = self._pool_hidden_states(
                hidden_states, 
                batch.get("attention_mask")
            )
            logits = self.classifier(pooled_output)
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

    def evaluate(self) -> Dict[str, float]:
        """Enhanced evaluation with classification metrics - Overrides parent"""
        if self.eval_dataloader is None:
            return {}
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        for batch in self.eval_dataloader:
            step_results = self.evaluation_step(batch)
            total_loss += step_results["loss"]
            total_correct += step_results["correct"]
            total_samples += step_results["total"]
            all_predictions.extend(step_results["predictions"])
            all_targets.extend(step_results["targets"])
        
        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        
        # Enhanced classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        self.latest_eval_loss = avg_loss
        self.all_predictions = all_predictions
        self.all_targets = all_targets
        
        # Log detailed classification metrics
        self._log_classification_metrics(all_targets, all_predictions)
        
        return {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1
        }

    def _log_classification_metrics(self, targets: List[int], predictions: List[int]):
        """Classification-specific metrics logging"""
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
                
                # Metrics table
                metrics_table = wandb.Table(columns=["Metric", "Value"])
                metrics_table.add_data("Accuracy", class_report['accuracy'])
                metrics_table.add_data("Weighted F1", class_report['weighted avg']['f1-score'])
                metrics_table.add_data("Macro F1", class_report['macro avg']['f1-score'])
                
                wandb.log({"classification_metrics": metrics_table}, step=self.global_step)
                
            except Exception as e:
                logger.warning(f"WandB logging failed: {e}")
        
        # Console logging
        logger.info(f"Classification Metrics - Accuracy: {class_report['accuracy']:.4f}, "
                   f"F1: {class_report['weighted avg']['f1-score']:.4f}")

    def setup_data(self, train_dataloader=None, eval_dataloader=None):
        """Setup training and validation data - REQUIRED by BaseTrainer"""
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        if eval_dataloader is not None:
            self.eval_dataloader = eval_dataloader
        return self.train_dataloader, self.eval_dataloader

    def save_checkpoint(self, checkpoint_dir: Optional[str] = None, is_best: bool = False):
        """Save checkpoint including classifier head - Extends parent method"""
        # First call parent to save base model and training state
        checkpoint_dir = super().save_checkpoint(checkpoint_dir, is_best)
        
        # Additional: Save classifier head
        if self.classifier:
            classifier_path = os.path.join(checkpoint_dir, "classifier.bin")
            torch.save(self.classifier.state_dict(), classifier_path)
            logger.info(f"Classifier head saved to {classifier_path}")
        
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir: str):
        """Load checkpoint including classifier head - Extends parent method"""
        # First call parent to load base model and training state
        super().load_checkpoint(checkpoint_dir)
        
        # Additional: Load classifier head
        classifier_path = os.path.join(checkpoint_dir, "classifier.bin")
        if os.path.exists(classifier_path) and self.classifier:
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
            logger.info(f"Classifier head loaded from {classifier_path}")

    def train(self):
        """Enhanced training loop with classification-specific progress - Uses parent utilities"""
        if self.train_dataloader is None:
            logger.warning("No training data available")
            return

        self.setup_wandb()
        
        from rich.console import Console
        console = Console()
        console.rule(f"[bold green]Starting SFT Classification Training for {self.config.num_epochs} epochs")
        console.print(f"[bold blue]Classes: {self.class_names}[/bold blue]")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # SIMPLIFIED progress bar - avoid complex field dependencies
            progress = create_progress_bar(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            task = progress.add_task(
                f"Epoch {epoch + 1}/{self.config.num_epochs}",
                total=len(self.train_dataloader)
            )

            with progress:
                epoch_loss, num_steps = self.training_flow.run_epoch(epoch, progress, task)

            self.training_flow.handle_end_of_epoch(epoch)

        # Enhanced summary with classification metrics
        summary_table = create_training_summary_table(
            "SFT Classification Training Summary", 
            self.config, 
            self.latest_eval_loss, 
            self.best_checkpoint_path, 
            self.wandb_url
        )
        
        # Add classification-specific metrics
        if hasattr(self, 'all_predictions') and self.all_predictions:
            accuracy = accuracy_score(self.all_targets, self.all_predictions)
            summary_table.add_row("Final Accuracy", f"{accuracy:.4f}")
        
        console.print(summary_table)
        print_training_completion("SFT Classification Training Completed!")
        
        if wandb.run is not None:
            wandb.finish()

    def debug_model_outputs(self, input_ids: torch.Tensor):
        """
        Debug method to understand what the model returns
        """
        print(f"\n🔍 DEBUG MODEL OUTPUTS:")
        print(f"Input shape: {input_ids.shape}")
        
        # Test 1: Direct model call
        print(f"\n1. Direct model call:")
        try:
            direct_output = self.model(input_ids)
            print(f"   ✅ Direct output shape: {direct_output.shape}")
            print(f"   ✅ Direct output type: {type(direct_output)}")
        except Exception as e:
            print(f"   ❌ Direct call failed: {e}")
        
        # Test 2: Check model architecture
        print(f"\n2. Model architecture:")
        print(f"   Model type: {type(self.model)}")
        print(f"   Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')][:10]}...")
        
        if hasattr(self.model, 'transformer'):
            print(f"   ✅ Has 'transformer' attribute")
        if hasattr(self.model, 'h'):
            print(f"   ✅ Has 'h' attribute") 
        if hasattr(self.model, 'wte'):
            print(f"   ✅ Has 'wte' attribute")
        if hasattr(self.model, 'wpe'):
            print(f"   ✅ Has 'wpe' attribute")
        if hasattr(self.model, 'ln_f'):
            print(f"   ✅ Has 'ln_f' attribute")
        
        # Test 3: Try hidden states extraction
        print(f"\n3. Hidden states extraction:")
        try:
            hidden_states = self._get_hidden_states(input_ids)
            print(f"   ✅ Hidden states shape: {hidden_states.shape}")
            
            pooled = self._pool_hidden_states(hidden_states)
            print(f"   ✅ Pooled shape: {pooled.shape}")
            
            logits = self.classifier(pooled)
            print(f"   ✅ Classifier logits shape: {logits.shape}")
            print(f"   🎉 SUCCESS: All shapes compatible!")
            
        except Exception as e:
            print(f"   ❌ Hidden states extraction failed: {e}")

    def update_best_metric(self, eval_results, key: str = None):
        """Override base method to use classifier-specific metric"""
        if key is None:
            key = self.config.metric_for_best_model
        return super().update_best_metric(eval_results, key)