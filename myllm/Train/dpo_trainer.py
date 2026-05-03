import copy
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

import wandb

from .base_trainer import BaseTrainer
from .utils.progress_utils import create_progress_bar
from .utils.summary_utils import create_training_summary_table, print_training_completion

logger = logging.getLogger(__name__)


class DPOTrainer(BaseTrainer):
    """
    Direct Preference Optimization trainer.

    Expects batches with keys:
        chosen        (B, T)  — preferred response token ids
        chosen_mask   (B, T)  — 1 for real tokens, 0 for padding
        rejected      (B, T)  — dispreferred response token ids
        rejected_mask (B, T)  — 1 for real tokens, 0 for padding

    Reference: Rafailov et al. 2023 — https://arxiv.org/abs/2305.18290
    """

    def __init__(self, config, model_config=None, model=None):
        super().__init__(config, model_config, model)
        self.reference_model: Optional[nn.Module] = None
        self.beta: float = getattr(config, "beta", 0.1)

    # ── Model setup ───────────────────────────────────────────────────────────

    def setup_model(self) -> nn.Module:
        """Build policy model, then freeze a deep copy as the reference."""
        from myllm.api import LLM
        from myllm.Train.utils.model_utils import load_pretrained_weights, setup_model_compilation

        if self.model is not None:
            logger.info("Using externally provided policy model.")
            self.model.to(self.device)
        else:
            if self.model_config is None:
                raise ValueError("model_config required to create a model.")
            logger.info(f"Creating policy model: {self.model_config.name}")
            llm = LLM(config=self.model_config, device=self.device)
            load_pretrained_weights(
                llm,
                getattr(self.config, "pretrained_variant", None),
                "gpt2",
            )
            self.model = llm.model
            self.model = setup_model_compilation(
                self.model,
                getattr(self.config, "use_compile", False),
                self.config,
            )

        self.model.to(self.device)
        self.setup_tokenizer()

        # Frozen deep copy — never updated
        self.reference_model = copy.deepcopy(self.model)
        self.reference_model.to(self.device)
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad_(False)

        logger.info("Reference model frozen.")
        return self.model

    # ── Data setup ────────────────────────────────────────────────────────────

    def setup_data(self, train_dataloader=None, eval_dataloader=None):
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        if eval_dataloader is not None:
            self.eval_dataloader = eval_dataloader
        return self.train_dataloader, self.eval_dataloader

    # ── Batch helpers (required by BaseTrainer ABC) ───────────────────────────

    def _prepare_batch(self, batch: Dict) -> Dict:
        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _get_labels(self, batch: Dict) -> torch.Tensor:
        # DPO doesn't use a label tensor — return chosen ids as a no-op placeholder
        return batch["chosen"]

    # ── Core DPO maths ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_logprobs(
        logits: torch.Tensor,          # (B, T, V)
        labels: torch.Tensor,          # (B, T)
        mask: Optional[torch.Tensor],  # (B, T) — 1 = real token
    ) -> torch.Tensor:                 # (B,)   — per-sequence mean log-prob
        """Compute per-sequence average log probability of the label tokens."""
        # Shift: predict token i+1 from position i
        labels = labels[:, 1:].clone()          # (B, T-1)
        logits = logits[:, :-1, :]              # (B, T-1, V)

        log_probs = F.log_softmax(logits, dim=-1)  # (B, T-1, V)

        # Gather log prob of the actual next token at each position
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)                           # (B, T-1)

        if mask is not None:
            mask = mask[:, 1:].clone().float()  # align with shifted labels
            token_log_probs = token_log_probs * mask
            return token_log_probs.sum(-1) / mask.sum(-1).clamp(min=1)
        return token_log_probs.mean(-1)

    def _dpo_loss(
        self,
        pi_chosen:   torch.Tensor,  # (B,)
        pi_rejected: torch.Tensor,  # (B,)
        ref_chosen:  torch.Tensor,  # (B,)
        ref_rejected: torch.Tensor, # (B,)
    ) -> tuple:
        """
        L = -log σ( β * ((log π(y_w|x) - log π_ref(y_w|x))
                        - (log π(y_l|x) - log π_ref(y_l|x))) )
        """
        logits = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
        loss   = -F.logsigmoid(self.beta * logits).mean()

        chosen_rewards   = (pi_chosen  - ref_chosen).detach().mean()
        rejected_rewards = (pi_rejected - ref_rejected).detach().mean()
        return loss, chosen_rewards, rejected_rewards

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, batch: Dict) -> Dict[str, Any]:
        """One DPO gradient step. Returns loss + reward metrics."""
        self.model.train()
        batch = self._prepare_batch(batch)

        chosen   = batch["chosen"]
        rejected = batch["rejected"]
        c_mask   = batch.get("chosen_mask")
        r_mask   = batch.get("rejected_mask")

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            # Policy forward passes
            pi_logits_chosen   = self.model(chosen)    # (B, T, V)
            pi_logits_rejected = self.model(rejected)

            pi_chosen   = self._compute_logprobs(pi_logits_chosen,   chosen,   c_mask)
            pi_rejected = self._compute_logprobs(pi_logits_rejected, rejected, r_mask)

            # Reference forward passes (no grad)
            with torch.no_grad():
                ref_chosen   = self._compute_logprobs(
                    self.reference_model(chosen),   chosen,   c_mask)
                ref_rejected = self._compute_logprobs(
                    self.reference_model(rejected), rejected, r_mask)

            loss, chosen_rewards, rejected_rewards = self._dpo_loss(
                pi_chosen, pi_rejected, ref_chosen, ref_rejected
            )
            scaled_loss = loss / max(1, self.config.gradient_accumulation_steps)

        self.scaler.scale(scaled_loss).backward()

        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()

        return {
            "loss":             float(loss.item()),
            "chosen_rewards":   float(chosen_rewards.item()),
            "rejected_rewards": float(rejected_rewards.item()),
            "reward_margin":    float((chosen_rewards - rejected_rewards).item()),
        }

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        totals = dict(loss=0., chosen=0., rejected=0., n=0)

        for batch in self.eval_dataloader:
            batch = self._prepare_batch(batch)
            with torch.no_grad(), autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu"
            ):
                c_mask = batch.get("chosen_mask")
                r_mask = batch.get("rejected_mask")

                pi_c  = self._compute_logprobs(self.model(batch["chosen"]),   batch["chosen"],   c_mask)
                pi_r  = self._compute_logprobs(self.model(batch["rejected"]), batch["rejected"], r_mask)
                ref_c = self._compute_logprobs(self.reference_model(batch["chosen"]),   batch["chosen"],   c_mask)
                ref_r = self._compute_logprobs(self.reference_model(batch["rejected"]), batch["rejected"], r_mask)

                loss, c_rew, r_rew = self._dpo_loss(pi_c, pi_r, ref_c, ref_r)

            b = batch["chosen"].size(0)
            totals["loss"]     += loss.item()     * b
            totals["chosen"]   += c_rew.item()    * b
            totals["rejected"] += r_rew.item()    * b
            totals["n"]        += b

        n = max(totals["n"], 1)
        metrics = {
            "eval_loss":             totals["loss"]     / n,
            "eval_chosen_rewards":   totals["chosen"]   / n,
            "eval_rejected_rewards": totals["rejected"] / n,
            "eval_reward_margin":    (totals["chosen"] - totals["rejected"]) / n,
        }
        self.latest_eval_loss = metrics["eval_loss"]
        return metrics

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self):
        if self.train_dataloader is None:
            logger.warning("No training data. Call setup_data() first.")
            return

        self.setup_wandb()

        from rich.console import Console
        console = Console()
        console.rule(f"[bold green]DPO Training — β={self.beta}  epochs={self.config.num_epochs}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            progress = create_progress_bar(f"Epoch {epoch+1}/{self.config.num_epochs}")
            task = progress.add_task(
                f"Epoch {epoch+1}/{self.config.num_epochs}",
                total=len(self.train_dataloader),
                loss=0., margin=0., eval_loss=0.,
            )

            epoch_loss = 0.
            with progress:
                for batch in self.train_dataloader:
                    metrics = self.training_step(batch)
                    self.global_step += 1
                    epoch_loss += metrics["loss"]

                    progress.update(
                        task,
                        advance=1,
                        loss=round(metrics["loss"], 4),
                        margin=round(metrics["reward_margin"], 4),
                    )

                    if self.should_log():
                        self.log_metrics({
                            "train/loss":          metrics["loss"],
                            "train/chosen_reward": metrics["chosen_rewards"],
                            "train/rejected_reward": metrics["rejected_rewards"],
                            "train/reward_margin": metrics["reward_margin"],
                        })

                    if (self.should_evaluate() and
                            self.global_step % self.config.eval_steps == 0):
                        eval_metrics = self.evaluate()
                        self.log_metrics(eval_metrics)
                        progress.update(task, eval_loss=round(eval_metrics.get("eval_loss", 0), 4))
                        self.model.train()

                    if (self.should_save_checkpoint() and
                            self.global_step % self.config.save_steps == 0):
                        is_best = self.update_best_metric({"eval_loss": self.latest_eval_loss or 0})
                        self.save_checkpoint(is_best=is_best)

            avg = epoch_loss / max(len(self.train_dataloader), 1)
            logger.info(f"Epoch {epoch+1} avg loss: {avg:.4f}")

            eval_metrics = self.evaluate()
            if eval_metrics:
                self.log_metrics(eval_metrics)
                logger.info(f"Eval — loss={eval_metrics['eval_loss']:.4f}  "
                            f"margin={eval_metrics['eval_reward_margin']:.4f}")

        summary = create_training_summary_table(
            "DPO Training Summary",
            self.config,
            self.latest_eval_loss,
            self.best_checkpoint_path,
            self.wandb_url,
        )
        console.print(summary)
        print_training_completion("DPO Training Completed!")

        if wandb.run is not None:
            wandb.finish()
