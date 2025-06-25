"""
GPT Model API Wrapper

A simple, clean API interface for your GPT transformer model.
Provides easy-to-use methods for loading, training, fine-tuning, and text generation.

Example Usage:
    # Initialize API
    api = GPTAPI("gpt2-small")
    
    # Train from scratch
    api.train_from_scratch("train.txt", epochs=10)
    
    # Load and fine-tune
    api.load_model("checkpoint.pt")
    api.finetune("finetune_data.txt", epochs=5)
    
    # Generate text
    text = api.generate("Hello world", max_length=100)
    print(text)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

# Import your existing modules
from model import GPT
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for training."""
    
    def __init__(self, file_path: str, block_size: int = 1024, vocab_size: int = 50304):
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Read text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple character-level tokenization
        chars = sorted(list(set(text)))
        self.vocab_size = min(len(chars), vocab_size)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars[:self.vocab_size])}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars[:self.vocab_size])}
        
        # Tokenize text
        self.tokens = [self.char_to_idx.get(ch, 0) for ch in text]
        
        logger.info(f"Dataset loaded: {len(self.tokens)} tokens, vocab size: {len(self.char_to_idx)}")
    
    def __len__(self):
        return max(1, len(self.tokens) // self.block_size)
    
    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        
        if end > len(self.tokens):
            # Handle last batch
            chunk = self.tokens[start:] + [0] * (end - len(self.tokens))
        else:
            chunk = self.tokens[start:end]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class GPTAPI:
    """
    Main API class for GPT model operations.
    """
    
    def __init__(self, config_name: str = "gpt2-small", device: Optional[str] = None):
        """
        Initialize the GPT API.
        
        Args:
            config_name: Name of the configuration to use
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.config = Config.from_name(config_name)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None  # Will be set when loading dataset
        
        logger.info(f"Initialized GPTAPI with config: {config_name}, device: {self.device}")
    
    def create_model(self) -> GPT:
        """Create a new model instance."""
        model = GPT(self.config).to(self.device)
        
        # Initialize weights
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        model.apply(_init_weights)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
        
        return model
    
    def load_model(self, checkpoint_path: str) -> None:
        """
        Load a model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load config if available
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        
        # Create model and load state
        self.model = self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load tokenizer if available
        if 'tokenizer' in checkpoint:
            self.tokenizer = checkpoint['tokenizer']
        
        logger.info(f"Model loaded from {checkpoint_path}")
    
    def save_model(self, checkpoint_path: str, optimizer: Optional[optim.Optimizer] = None, 
                   epoch: int = 0, loss: float = 0.0) -> None:
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save the checkpoint
            optimizer: Optimizer state to save
            epoch: Current epoch number
            loss: Current loss value
        """
        if self.model is None:
            raise ValueError("No model to save. Create or load a model first.")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'loss': loss,
            'tokenizer': self.tokenizer
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")
    
    def train_from_scratch(
        self,
        train_data: str,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        save_every: int = 1000,
        eval_every: int = 500,
        output_dir: str = "./checkpoints",
        val_data: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train model from scratch.
        
        Args:
            train_data: Path to training data file
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            save_every: Save checkpoint every N steps
            eval_every: Evaluate every N steps
            output_dir: Directory to save checkpoints
            val_data: Path to validation data file (optional)
        
        Returns:
            Dictionary containing training history
        """
        # Create model
        self.model = self.create_model()
        
        # Create dataset and dataloader
        dataset = TextDataset(train_data, self.config.block_size, self.config.vocab_size)
        self.tokenizer = {'char_to_idx': dataset.char_to_idx, 'idx_to_char': dataset.idx_to_char}
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Validation dataset if provided
        val_dataloader = None
        if val_data:
            val_dataset = TextDataset(val_data, self.config.block_size, self.config.vocab_size)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.1)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'steps': []}
        step = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                step += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
                # Evaluate and save
                if step % eval_every == 0:
                    val_loss = self._evaluate(val_dataloader) if val_dataloader else None
                    history['train_loss'].append(loss.item())
                    history['val_loss'].append(val_loss)
                    history['steps'].append(step)
                    
                    if val_loss:
                        logger.info(f"Step {step}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")
                
                if step % save_every == 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
                    self.save_model(checkpoint_path, optimizer, epoch, loss.item())
            
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save final model
        final_path = os.path.join(output_dir, "final_model.pt")
        self.save_model(final_path, optimizer, epochs, avg_loss)
        
        return history
    
    def finetune(
        self,
        train_data: str,
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        save_every: int = 500,
        output_dir: str = "./finetune_checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Fine-tune existing model.
        
        Args:
            train_data: Path to fine-tuning data
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate (usually lower than training from scratch)
            save_every: Save frequency
            output_dir: Output directory
        
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first using load_model()")
        
        # Create dataset
        dataset = TextDataset(train_data, self.config.block_size, self.config.vocab_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer with lower learning rate
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Training loop
        history = {'train_loss': [], 'steps': []}
        step = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        for epoch in range(epochs):
            self.model.train()
            
            pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch+1}/{epochs}")
            
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                step += 1
                history['train_loss'].append(loss.item())
                history['steps'].append(step)
                
                pbar.set_postfix({'loss': loss.item()})
                
                if step % save_every == 0:
                    checkpoint_path = os.path.join(output_dir, f"finetune_step_{step}.pt")
                    self.save_model(checkpoint_path, optimizer, epoch, loss.item())
        
        # Save final finetuned model
        final_path = os.path.join(output_dir, "finetuned_model.pt")
        self.save_model(final_path, optimizer, epochs, loss.item())
        
        return history
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from the model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to use sampling or greedy decoding
        
        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
        
        if self.tokenizer is None:
            raise ValueError("No tokenizer available. Train or load a model with tokenizer first.")
        
        self.model.eval()
        
        # Tokenize prompt
        char_to_idx = self.tokenizer['char_to_idx']
        idx_to_char = self.tokenizer['idx_to_char']
        
        # Handle unknown characters
        input_ids = []
        for char in prompt:
            if char in char_to_idx:
                input_ids.append(char_to_idx[char])
            else:
                input_ids.append(0)  # Use 0 for unknown characters
        
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Initialize KV cache for faster generation
        self.model.initialize_kv_cache(batch_size=1, max_seq_len=max_length + len(input_ids[0]))
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for the last token
                logits = self.model(generated_ids[:, -1:], use_cache=True)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, indices, values)
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop if we generate an end token (if you have one)
                # if next_token.item() == end_token_id:
                #     break
        
        # Reset cache
        self.model.reset_cache()
        
        # Decode generated text
        generated_text = ""
        for token_id in generated_ids[0].cpu().numpy():
            if token_id in idx_to_char:
                generated_text += idx_to_char[token_id]
        
        return generated_text
    
    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation data."""
        if dataloader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.model is None:
            return {"error": "No model loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "config": self.config.__dict__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": self.device,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize API
    api = GPTAPI("gpt2-small")
    
    # Print model info
    print("Model Info:")
    print(json.dumps(api.get_model_info(), indent=2))
    
    # Example: Train from scratch (uncomment to use)
    # history = api.train_from_scratch(
    #     train_data="train.txt",
    #     epochs=2,
    #     batch_size=4,
    #     learning_rate=3e-4
    # )
    
    # Example: Load and generate (uncomment to use)
    # api.load_model("checkpoints/final_model.pt")
    # generated_text = api.generate("Hello", max_length=50)
    # print(f"Generated: {generated_text}")