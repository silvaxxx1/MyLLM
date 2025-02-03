# train_dist.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import argparse
import logging
from train import eval_model
import tiktoken
from dataloader import GPTDataLoader  
from model import GPTModel 
from configs.gpt_config import GPT_CONFIG_124M  # Import your configuration
from train_utils import plot_losses, save_model, load_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def trainerV4(model, train_loader, val_loader, optimizer, device, num_epochs,
              eval_freq, eval_iter, start_context, tokenizer, save_path, scheduler,
              total_batch, micro_batch, sequence_length, criterion):
    """
    Trains the model with distributed data parallel (DDP) using torchrun.
    """
    # Wrap model with DDP
    model = DDP(model, device_ids=[device])

    # Load checkpoint if it exists
    checkpoint_path = os.path.join(save_path, "latest_checkpoint.pt")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']

    grad_accum = total_batch // (micro_batch * sequence_length)
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Create a DistributedSampler for train_loader
    train_sampler = DistributedSampler(train_loader.dataset, shuffle=True)
    train_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size,
                              sampler=train_sampler)
    train_iterator = iter(train_loader)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        logging.info(f"[Epoch {epoch + 1}/{num_epochs}] Starting training epoch")

        # Loop through gradient accumulation steps
        for step in range(grad_accum):
            optimizer.zero_grad()
            loss_accum = 0

            # Loop through micro-batches for gradient accumulation
            for micro_step in range(micro_batch):
                try:
                    input_batch, target_batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    input_batch, target_batch = next(train_iterator)

                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_batch)

                loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1)) / grad_accum
                loss_accum += loss.item()
                loss.backward()

            if dist.get_rank() == 0:
                logging.info(f"Micro-batch {step + 1}/{grad_accum} completed.")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = eval_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                if dist.get_rank() == 0:
                    logging.info(f"[Evaluation] Step {global_step}: Train Loss = {train_loss}, Val Loss = {val_loss}")

            scheduler.step()

        if dist.get_rank() == 0:
            logging.info(f"[Epoch {epoch + 1}] completed with avg loss {loss_accum / grad_accum}")

        if (epoch + 1) % 5 == 0 and dist.get_rank() == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Checkpoint saved at epoch {epoch + 1}")

    return train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0004, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument("--eval_freq", type=int, default=5, help="Frequency of evaluations")
    parser.add_argument("--eval_iter", type=int, default=5, help="Iterations for evaluation")
    parser.add_argument("--total_batch", type=int, default=500000, help="Total batch size for training")
    parser.add_argument("--sequence_length", type=int, default=256, help="Sequence length for input data")
    parser.add_argument("--save_path", type=str, default="checkpoints", help="Path to save checkpoints")

    args = parser.parse_args()

    init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = local_rank
    torch.cuda.set_device(local_rank)

    model = GPTModel(GPT_CONFIG_124M)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.0001, last_epoch=-1)
    micro_batch = args.batch_size
    criterion = nn.CrossEntropyLoss()
    start_context = "Be Humble"
    train_dataloader = GPTDataLoader(args.train_file, args.max_len, args.stride, args.batch_size)
    val_dataloader = GPTDataLoader(args.val_file, args.max_len, args.stride, args.batch_size)
    tok = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, track_tokens_seen = trainerV4(
        model, train_dataloader, val_dataloader, optimizer, device, args.num_epochs,
        args.eval_freq, args.eval_iter, start_context, tokenizer=tok, save_path=args.save_path, scheduler=scheduler,
        total_batch=args.total_batch, micro_batch=micro_batch, sequence_length=args.sequence_length, criterion=criterion
    )

    destroy_process_group()
    logging.info("Training completed successfully.")
