# myllm/Train/sft_trainer.py
import os
import time
import logging
from typing import Dict, Any, Optional, List, Sequence

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, MofNCompleteColumn
)
from rich.table import Table

import wandb

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)
console = Console()


class SFTTrainer(BaseTrainer):
    """
    SFT Trainer with single-device and DDP readiness.
    Improvements implemented:
      - Token-level response-template masking (pre-tokenized)
      - Perplexity clamping
      - Resume from checkpoint
      - Early stopping
      - DDP hooks (init, wrap)
      - Gradient norm logging and parameter count
      - Robust torch.compile usage
      - Checkpoint filenames include step + timestamp
    """

    def __init__(self, config, model_config=None, model=None):
        super().__init__(config, model_config, model)

        # Data & model
        self.train_dataloader: Optional[DataLoader] = None
        self.eval_dataloader: Optional[DataLoader] = None
        self.model = model
        self.tokenizer = None

        # training state
        self.latest_eval_loss: Optional[float] = None
        self.best_checkpoint_path: Optional[str] = None
        self.wandb_url: str = "N/A"
        self.global_step: int = getattr(self, "global_step", 0)
        self.current_epoch: int = 0

        # AMP
        self.use_cuda = torch.cuda.is_available() and not getattr(config, "force_cpu", False)
        self.scaler = GradScaler()

        # SFT specifics
        self.mask_instructions: bool = getattr(config, "mask_instructions", True)
        self.response_template: str = getattr(config, "response_template", "### Response:")
        self.response_template_ids: Optional[torch.Tensor] = None  # set in setup_model after tokenizer available

        # training utilities
        self.optimizer = None
        self.scheduler = None
        self.early_stop_patience = getattr(config, "early_stop_patience", None)
        self._no_improve_steps = 0

        # DDP flag
        self.is_ddp = False

    # ------------------------------
    # DDP helpers
    # ------------------------------
    def init_distributed(self, world_size: int, rank: int, backend: str = "nccl"):
        """
        Initialize torch.distributed. Call this *before* setup_model when running DDP.
        """
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        self.is_ddp = True
        logger.info(f"Initialized distributed: rank={rank} world_size={world_size}")

    def wrap_model_ddp(self):
        """
        Wrap model in DDP. Call after moving model to device.
        """
        if self.is_ddp:
            device = torch.device("cuda", dist.get_rank() % torch.cuda.device_count()) if self.use_cuda else None
            if device:
                torch.cuda.set_device(device)
                self.model.to(device)
                self.model = DDP(self.model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
            else:
                self.model = DDP(self.model)
            logger.info("Model wrapped in DDP")

    # ------------------------------
    # Model and Data Setup
    # ------------------------------
    def setup_model(self) -> nn.Module:
        """
        Build the model and tokenizer (if not provided) and pre-tokenize the response template.
        """
        if self.model is None:
            # user-provided GPT factory expected (same as before)
            from myllm.model import GPT
            from myllm.Tokenizers.factory import get_tokenizer
            from myllm.Tokenizers.wrapper import TokenizerWrapper

            logger.info(f"Creating model: {self.model_config.name}")
            self.model = GPT(self.model_config)

            tokenizer_raw = get_tokenizer(self.config.tokenizer_name)
            self.tokenizer = TokenizerWrapper(tokenizer_raw)

            # Ensure pad token id exists and is numeric
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                # sometimes wrapper stores pad_token as id or string; prefer eos id if present
                pad_id = getattr(self.tokenizer, "eos_token_id", None)
            self.tokenizer.pad_token = pad_id if pad_id is not None else 0

        # Move model to device
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model.to(self.device)

        # Attempt to compile model if requested
        if getattr(self.config, "use_compile", False):
            try:
                logger.info("Attempting torch.compile() optimization")
                self.model = torch.compile(self.model)
            except Exception as e:
                logger.warning(f"torch.compile() failed or unsupported: {e}")

        # Pre-tokenize response_template (token-level) and cache ids
        try:
            rt_ids = self.tokenizer.encode(self.response_template)
            # normalize to list of ints
            if torch.is_tensor(rt_ids):
                rt_ids = rt_ids.tolist()
            # tokenizers sometimes return nested list
            if isinstance(rt_ids, (list, tuple)) and len(rt_ids) == 1 and isinstance(rt_ids[0], (list, tuple)):
                rt_ids = list(rt_ids[0])
            rt_ids = [int(x) for x in rt_ids]
            if len(rt_ids) == 0:
                logger.warning("response_template tokenization returned empty list; fallback to heuristic masking")
                self.response_template_ids = None
            else:
                self.response_template_ids = torch.tensor(rt_ids, dtype=torch.long).to(self.device)
        except Exception as e:
            logger.warning(f"Failed to pre-tokenize response_template: {e}. Falling back to heuristic masking.")
            self.response_template_ids = None

        # If DDP requested, wrap model now (after moving to device)
        if self.is_ddp:
            self.wrap_model_ddp()

        return self.model

    def setup_data(self):
        # left for user to set dataloaders externally
        logger.info("Setting up SFT datasets (user must assign dataloaders).")

    def setup_optimizer(self):
        """
        Create optimizer and scheduler. Use trainer config LR (avoid model_config confusion).
        """
        if self.model is None:
            raise ValueError("Model must be setup before optimizer")

        lr = getattr(self.config, "learning_rate", None) or getattr(self.model_config, "learning_rate", 1e-4)

        # default to AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=getattr(self.model_config, "weight_decay", 0.0),
            betas=(getattr(self.model_config, "beta1", 0.9), getattr(self.model_config, "beta2", 0.999))
        )

        # simple scheduler example (linear warmup)
        if getattr(self.config, "scheduler_type", None) and getattr(self.config.scheduler_type, "value", "") == "linear":
            try:
                from torch.optim.lr_scheduler import LinearLR
                self.scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=max(1, int(getattr(self.config, "warmup_steps", 1))))
            except Exception:
                logger.warning("LinearLR unavailable; skipping scheduler setup.")

        # log parameter count
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Optimizer set up. Model parameter count: {param_count:,}")

    # ------------------------------
    # Checkpointing (save, load, resume)
    # ------------------------------
    def _checkpoint_name(self, step: Optional[int] = None):
        ts = int(time.time())
        step = step if step is not None else getattr(self, "global_step", 0)
        return f"checkpoint-step-{step}-{ts}.pt"

    def save_checkpoint(self, out_path: Optional[str] = None, is_best: bool = False) -> str:
        out_path = out_path or os.path.join(getattr(self.config, "output_dir", "."), self._checkpoint_name())
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # handle DDP wrapped model (unwrap)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        save_dict = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "global_step": getattr(self, "global_step", 0),
            "epoch": getattr(self, "current_epoch", 0),
            "config": vars(self.config) if not isinstance(self.config, dict) else self.config
        }

        torch.save(save_dict, out_path)
        logger.info(f"Saved checkpoint: {out_path}")

        if is_best:
            self.best_checkpoint_path = out_path

        return out_path

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device)
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        if load_optimizer and self.optimizer and ckpt.get("optimizer_state_dict") is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")
        self.global_step = ckpt.get("global_step", self.global_step)
        self.current_epoch = ckpt.get("epoch", self.current_epoch)
        logger.info(f"Loaded checkpoint from {path} (step={self.global_step}, epoch={self.current_epoch})")
        return ckpt

    # ------------------------------
    # Label creation: token-level masking (fast)
    # ------------------------------
    def create_sft_labels(self, batch) -> torch.Tensor:
        """
        Create labels for SFT training using token-level search for the response template.
        Falls back to heuristic token ratio masking if template tokens are unavailable.

        Input:
          batch: dict with at least "input_ids": Tensor[B, L]

        Returns:
          labels: Tensor[B, L-1] shifted for next-token prediction with -100 masked positions.
        """
        input_ids: torch.Tensor = batch["input_ids"]
        labels = input_ids.clone()
        if not self.mask_instructions:
            return labels[:, 1:].contiguous()

        B, L = input_ids.shape

        # fast token-level search if pre-tokenized template exists
        if self.response_template_ids is not None:
            rt = self.response_template_ids.to(input_ids.device)
            m = rt.size(0)
            for i in range(B):
                seq = input_ids[i]
                found = False
                # sliding window search - pure tensor compares
                # use vectorized check where possible
                if m <= 0 or m > L:
                    labels[i, : L // 2] = -100
                    continue
                # naive loop (acceptable for moderate sequence lengths)
                for j in range(0, L - m + 1):
                    if torch.equal(seq[j : j + m], rt):
                        # mask everything before and including template tokens
                        labels[i, : j + m] = -100
                        found = True
                        break
                if not found:
                    labels[i, : L // 2] = -100
        else:
            # fallback: decode once per example (slower) and use text ratio to estimate token position
            for i in range(B):
                seq = input_ids[i]
                try:
                    text = self.tokenizer.decode(seq.tolist())
                    response_start = text.find(self.response_template)
                    if response_start != -1:
                        pre_len = response_start + len(self.response_template)
                        token_pos = int((pre_len / max(1, len(text))) * L)
                        labels[i, : token_pos] = -100
                    else:
                        labels[i, : L // 2] = -100
                except Exception:
                    labels[i, : L // 2] = -100

        # shift for next-token prediction
        return labels[:, 1:].contiguous()

    # ------------------------------
    # Training step
    # ------------------------------
    def train_step(self, batch) -> Dict[str, Any]:
        """
        Single forward/backward step with AMP, gradient accumulation, and clipping.
        """
        self.model.train()
        # tensors to device
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        device_type = "cuda" if self.use_cuda else "cpu"
        with autocast(device_type=device_type):
            # assume model returns logits of shape (B, L, V) or a dict; adapt to your GPT's forward signature
            logits = self.model(batch["input_ids"])
            # if model returns tuple/ModelOutput adjust accordingly in your codebase
            if hasattr(logits, "logits"):
                # huggingface-style ModelOutput
                logits = logits.logits

            labels = self.create_sft_labels(batch)

            # align shapes
            if logits.size(1) != labels.size(1):
                min_len = min(logits.size(1), labels.size(1))
                logits = logits[:, :min_len].contiguous()
                labels = labels[:, :min_len].contiguous()

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            # scale for grad accumulation
            grad_acc = max(1, getattr(self.config, "gradient_accumulation_steps", 1))
            if grad_acc > 1:
                loss = loss / grad_acc

        # backward with scaler
        self.scaler.scale(loss).backward()

        # gradient clipping (unscale first)
        if getattr(self.config, "max_grad_norm", 0) > 0 and self.optimizer is not None:
            try:
                self.scaler.unscale_(self.optimizer)
            except Exception:
                pass
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.config, "max_grad_norm", 1.0))
        else:
            grad_norm = None

        # optimizer step when accumulation boundary reached
        if (self.global_step + 1) % grad_acc == 0:
            try:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            except Exception as e:
                logger.exception(f"Optimizer step failed: {e}")

            if self.scheduler:
                try:
                    self.scheduler.step()
                except Exception:
                    pass

            # zero grads after step
            self.optimizer.zero_grad()

        # metrics
        return {"loss": loss.item(), "grad_norm": float(grad_norm) if grad_norm is not None else None}

    # ------------------------------
    # Evaluation
    # ------------------------------
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on eval_dataloader and compute token-normalized loss and perplexity.
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        device_type = "cuda" if self.use_cuda else "cpu"
        with torch.no_grad(), autocast(device_type=device_type):
            for batch in self.eval_dataloader:
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                logits = self.model(batch["input_ids"])
                if hasattr(logits, "logits"):
                    logits = logits.logits

                labels = self.create_sft_labels(batch)

                if logits.size(1) != labels.size(1):
                    min_len = min(logits.size(1), labels.size(1))
                    logits = logits[:, :min_len].contiguous()
                    labels = labels[:, :min_len].contiguous()

                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )

                valid_tokens = (labels != -100).sum().item()

                total_loss += loss.item()
                total_tokens += valid_tokens

        if total_tokens == 0:
            return {}

        avg_loss = total_loss / total_tokens
        avg_loss_clamped = float(max(min(avg_loss, 50.0), -50.0))
        perplexity = float(torch.exp(torch.tensor(avg_loss_clamped)).item())

        self.latest_eval_loss = avg_loss
        return {"eval_loss": avg_loss, "perplexity": perplexity}

    # ------------------------------
    # Training loop
    # ------------------------------
    def train(self):
        """
        Training loop with:
          - AMP
          - Gradient accumulation
          - Periodic evaluation & checkpointing
          - Early stopping
          - W&B logging (if configured)
        """
        if self.train_dataloader is None:
            logger.warning("No training data available")
            return

        # W&B
        try:
            self.wandb_url = wandb.run.url if wandb.run else "N/A"
        except Exception:
            self.wandb_url = "N/A"

        console.rule(f"[bold green]Starting SFT training for {self.config.num_epochs} epochs")
        console.print(f"[yellow]Instruction masking: {self.mask_instructions}")
        console.print(f"[yellow]Response template: '{self.response_template}'")

        # zero grads at epoch start to initialize accumulation properly
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            num_steps = 0

            # progress bar
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[yellow]LR: {task.fields[lr]:.2e}"),
                TextColumn("[red]Train Loss: {task.fields[train_loss]:.4f}"),
                TextColumn("[cyan]Avg Loss: {task.fields[avg_loss]:.4f}"),
                TextColumn("[magenta]Eval Loss: {task.fields[eval_loss]:.4f}"),
                TextColumn("[green]PPL: {task.fields[perplexity]:.2f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            )

            task = progress.add_task(
                f"SFT Epoch {epoch + 1}/{self.config.num_epochs}",
                total=len(self.train_dataloader),
                train_loss=0.0,
                avg_loss=0.0,
                eval_loss=0.0,
                perplexity=0.0,
                lr=0.0
            )

            with progress:
                for step_idx, batch in enumerate(self.train_dataloader):
                    step_results = self.train_step(batch)
                    epoch_loss += step_results["loss"]
                    num_steps += 1
                    self.global_step += 1

                    avg_loss = epoch_loss / num_steps
                    lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
                    eval_loss = self.latest_eval_loss or 0.0
                    perplexity = float(torch.exp(torch.clamp(torch.tensor(eval_loss), max=50)).item()) if eval_loss > 0 else 0.0

                    # update progress bar
                    progress.update(
                        task,
                        advance=1,
                        train_loss=step_results['loss'],
                        avg_loss=avg_loss,
                        eval_loss=eval_loss,
                        perplexity=perplexity,
                        lr=lr
                    )

                    # log to W&B
                    if self.should_log():
                        metrics = {
                            "train/loss": step_results["loss"],
                            "train/epoch_avg_loss": avg_loss,
                            "train/epoch": epoch,
                            "train/step": self.global_step,
                            "train/perplexity": float(torch.exp(torch.clamp(torch.tensor(avg_loss), max=50)).item())
                        }
                        if step_results.get("grad_norm") is not None:
                            metrics["train/grad_norm"] = step_results["grad_norm"]
                        # optionally include param count
                        metrics["model/param_count"] = sum(p.numel() for p in self.model.parameters())
                        if self.latest_eval_loss is not None:
                            metrics["eval/loss"] = self.latest_eval_loss
                            metrics["eval/perplexity"] = float(torch.exp(torch.clamp(torch.tensor(self.latest_eval_loss), max=50)).item())
                        self.log_metrics(metrics)

                    # periodic evaluation
                    if self.should_evaluate() and self.eval_dataloader is not None and self.global_step % max(1, getattr(self.config, "eval_steps", 1)) == 0:
                        eval_results = self.evaluate()
                        if eval_results:
                            eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
                            self.log_metrics(eval_metrics)
                            console.print(f"[bold magenta]Step {self.global_step} Eval Loss: {eval_results['eval_loss']:.4f}, Perplexity: {eval_results.get('perplexity', 0):.2f}[/bold magenta]")

                            is_best = self.update_best_metric(eval_results)
                            if is_best:
                                self.best_checkpoint_path = self.save_checkpoint(is_best=True)
                                self._no_improve_steps = 0
                            else:
                                self._no_improve_steps += 1

                            # early stopping check
                            if self.early_stop_patience is not None and self._no_improve_steps >= self.early_stop_patience:
                                console.print(f"[red]Early stopping triggered (no improvement for {self.early_stop_patience} evals).[/red]")
                                return

                    # periodic checkpoint save
                    if self.should_save_checkpoint() and self.global_step % max(1, getattr(self.config, "save_steps", 1)) == 0:
                        self.save_checkpoint()

            # end-of-epoch evaluation
            if self.eval_dataloader is not None:
                eval_results = self.evaluate()
                if eval_results:
                    eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
                    eval_metrics["eval/epoch"] = epoch
                    self.log_metrics(eval_metrics)
                    console.print(f"[bold magenta][End of Epoch {epoch + 1}] Eval Loss: {eval_results['eval_loss']:.4f}, Perplexity: {eval_results.get('perplexity', 0):.2f}[/bold magenta]")

                    is_best = self.update_best_metric(eval_results)
                    if is_best:
                        self.best_checkpoint_path = self.save_checkpoint(is_best=True)

        # summary
        table = Table(title="SFT Training Summary", show_lines=True)
        table.add_column("Epochs", justify="center")
        table.add_column("Final Eval Loss", justify="center")
        table.add_column("Final Perplexity", justify="center")
        table.add_column("Best Checkpoint", justify="center")
        table.add_column("W&B Run URL", justify="center")

        final_ppl = float(torch.exp(torch.clamp(torch.tensor(self.latest_eval_loss if self.latest_eval_loss is not None else 0.0), max=50)).item()) if self.latest_eval_loss else 0.0

        table.add_row(
            str(self.config.num_epochs),
            f"{self.latest_eval_loss:.4f}" if self.latest_eval_loss else "N/A",
            f"{final_ppl:.2f}" if self.latest_eval_loss else "N/A",
            self.best_checkpoint_path if self.best_checkpoint_path else "N/A",
            self.wandb_url
        )

        console.print(table)
        console.rule("[bold green]SFT Training completed!")
