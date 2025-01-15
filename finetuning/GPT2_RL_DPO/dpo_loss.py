import torch
import torch.nn.functional as F

def compute_dpo_loss(
      model_chosen_logprobs,       # Log-probabilities of the human-preferred responses (from the policy model)
      model_rejected_logprobs,     # Log-probabilities of the human-dispreferred responses (from the policy model)
      reference_chosen_logprobs,   # Log-probabilities of the human-preferred responses (from the reference model)
      reference_rejected_logprobs, # Log-probabilities of the human-dispreferred responses (from the reference model)
      beta=0.1,                    # Scaling hyperparameter for sensitivity in DPO optimization
    ):


    # Compute the log-probability ratios for the model
    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    # Compute the log-probability ratios for the reference model
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs

    # Compute the logits for the DPO loss
    logits = model_logratios - reference_logratios

    # Compute the DPO loss using a logistic sigmoid function (negative log-sigmoid)
    # This corresponds to the negative log of the probability of preferring the chosen response
    losses = -F.logsigmoid(beta * logits)

    # Optional tracking: Compute rewards for the preferred and dispreferred responses
    # These are the log-probability differences between the model and reference for each type of response
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    # Return the mean loss over the batch and the mean rewards for tracking purposes
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

def compute_logprobs(logits, labels, selection_mask=None):


    # Labels are the inputs shifted by one:
    # This ensures that each label corresponds to the token predicted by the model
    # (i.e., the label for token at position `i` is compared to the prediction at position `i-1`).
    labels = labels[:, 1:].clone()

    # Truncate logits to match the number of tokens in the shifted labels.
    # This removes the last token's prediction since it doesn't have a corresponding label.
    logits = logits[:, :-1, :]

    # Compute the log probabilities over the vocabulary using softmax.
    # Shape: (batch_size, num_tokens - 1, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the actual labels.
    # `labels` indicates the correct token index for each position in the sequence.
    # `torch.gather` selects the log probability of the correct token for each position.
    # Shape of `selected_log_probs`: (batch_size, num_tokens - 1)
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)  # Add an extra dimension to match `log_probs`'s last dimension.
    ).squeeze(-1)  # Remove the extra dimension to restore the shape.

    if selection_mask is not None:
        # If a selection mask is provided, apply it to filter out irrelevant tokens
        # (e.g., padding tokens that should not contribute to the loss or metric).
        # Shift the mask to align with the shifted labels.
        mask = selection_mask[:, 1:].clone()

        # Apply the mask to zero out log probabilities for excluded tokens.
        selected_log_probs = selected_log_probs * mask

        # Compute the average log probability excluding the padding tokens:
        # - `sum(-1)` sums the log probabilities for each token in the sequence.
        # - `mask.sum(-1)` gives the number of non-padding tokens for each sequence.
        # The result is the average log probability for each sequence.
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)

        # Return the average log probability for each sequence in the batch.
        return avg_log_prob

    else:
        # If no mask is provided, simply return the mean log probability across all tokens.
        # This averages over the token dimension (-1) for each sequence in the batch.
        return selected_log_probs.mean(-1)

def compute_dpo_loss_batch(batch, policy_model, reference_model, beta):
    """Compute the DPO loss on an input batch"""

    # where policy_model(batch["chosen"]) are the logits
    policy_chosen_log_probas = compute_logprobs(
        logits=policy_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"]
    )
    policy_rejected_log_probas = compute_logprobs(
        logits=policy_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"]
    )

    with torch.no_grad():
        ref_chosen_log_probas = compute_logprobs(
            logits=reference_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )
        ref_rejected_log_probas = compute_logprobs(
            logits=reference_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"]
        )
    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta
    )
    return loss, chosen_rewards, rejected_rewards

def compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta, num_batches=None):
    """Apply compute_dpo_loss_batch to a whole data loader"""

    total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
    if len(data_loader) == 0:
        return float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()

        else:
            break

    # calculate average
    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    return total_loss, total_chosen_rewards, total_rejected_rewards

def evaluate_dpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):
    """Compute the DPO loss for the training and validation dataset"""

    policy_model.eval()
    with torch.no_grad():
        train_loss, train_chosen_rewards, train_rejected_rewards = compute_dpo_loss_loader(
            data_loader=train_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

        val_loss, val_chosen_rewards, val_rejected_rewards = compute_dpo_loss_loader(
            data_loader=val_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

    res = {
        "train_loss": train_loss,
        "train_chosen_reward": train_chosen_rewards,
        "train_rejected_reward": train_rejected_rewards,
        "val_loss": val_loss,
        "val_chosen_reward": val_chosen_rewards,
        "val_rejected_reward": val_rejected_rewards
    }

    policy_model.train()
    return res


