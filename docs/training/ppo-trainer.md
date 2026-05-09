# PPOTrainer

**File:** `myllm/Train/ppo_trainer.py`
**Import:** `from myllm.train import PPOTrainer`
**Status:** Stub — scaffolding only, no real training yet.

---

## Current state

`PPOTrainer` satisfies the `BaseTrainer` ABC but `_train_loop()` is a no-op.

---

## Planned implementation

### Algorithm — Proximal Policy Optimization for RLHF

PPO is the classical RLHF algorithm (used in InstructGPT, ChatGPT).
A reward model scores completions; PPO optimises the policy to maximise reward
while staying close to a reference policy via a KL penalty.

**Four models required:**
1. **Actor** (`π_θ`) — policy being trained (GPT with SFT init)
2. **Critic** — value function head on top of Actor; predicts expected reward
3. **Reference** (`π_ref`) — frozen SFT model; used for KL penalty
4. **Reward model** — trained to score responses; frozen during PPO

**PPO objective:**

```
L_PPO = E[
    min(
        r_t(θ) × A_t,
        clip(r_t(θ), 1-ε, 1+ε) × A_t
    )
] - β × KL(π_θ || π_ref)
```

Where:
- `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)` — probability ratio
- `A_t` — advantage estimate (from critic)
- `ε` — clip range (typically `0.2`)
- `β` — KL coefficient

### Training loop (planned)

```
for each PPO iteration:
    1. Rollout — sample completions from actor
    2. Score   — reward model scores completions
    3. Advantage estimation — critic + GAE
    4. PPO update — multiple epochs over the rollout buffer
    5. KL check — stop early if KL divergence too large
```

### Config (planned — `PPOTrainerConfig`)

| Field | Description |
|-------|-------------|
| `kl_coeff` | KL penalty weight |
| `clip_range` | PPO clip epsilon |
| `value_loss_coeff` | Critic loss weight |
| `reward_model_path` | Path to trained reward model |
| `reference_model_path` | Path to frozen SFT reference |
| `rollout_batch_size` | Samples per PPO iteration |
| `ppo_epochs` | Gradient steps per rollout |

### Reference reading

- [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al. 2022
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) — Schulman et al. 2017
