from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Literal
from pathlib import Path
import json

@dataclass
class TrainingConfig:
    """Base training configuration that works with your existing ModelConfig"""
    
    # Training Loop
    max_steps: int = 1000
    max_epochs: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    
    # Optimization
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning Rate Scheduling
    lr_scheduler: Literal["linear", "cosine", "constant", "polynomial"] = "linear"
    lr_warmup_type: Literal["linear", "cosine"] = "linear"
    
    # Mixed Precision
    fp16: bool = False
    bf16: bool = True
    fp16_opt_level: str = "O1"
    
    # Logging & Checkpointing
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "./outputs"
    logging_dir: Optional[str] = None
    save_total_limit: int = 2
    
    # Data
    max_seq_length: int = 2048
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = True
    
    # Evaluation
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Reproducibility
    seed: int = 42
    data_seed: int = 42
    
    # Additional parameters
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Model-specific overrides (if needed to override ModelConfig)
    model_config_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: Union[str, Path]):
        """Save config to JSON"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Union[str, Path]):
        """Load config from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

@dataclass
class SFTConfig(TrainingConfig):
    """Supervised Fine-tuning specific config"""
    
    # SFT specific
    packing: bool = False
    dataset_text_field: str = "text"
    max_seq_length: int = 2048
    
    # LoRA/Adapters
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: float = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_bias: str = "none"
    
    # QLoRA specific
    use_qlora: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Data formatting
    formatting_func: Optional[str] = None  # Name of formatting function
    response_template: str = "\n### Response:\n"
    instruction_template: str = "\n### Instruction:\n"

@dataclass 
class DPOConfig(TrainingConfig):
    """Direct Preference Optimization config"""
    
    # DPO specific
    beta: float = 0.1
    max_length: int = 2048
    max_prompt_length: int = 1024
    max_target_length: int = 1024
    
    # Reference model
    reference_free: bool = False
    ref_model_path: Optional[str] = None
    
    # Data format
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"
    prompt_key: str = "prompt"
    
    # Loss computation
    loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid"
    label_smoothing: float = 0.0

@dataclass
class PretrainConfig(TrainingConfig):
    """Pretraining specific config"""
    
    # Data
    dataset_config_name: Optional[str] = None
    streaming: bool = False
    
    # Masking (for MLM if needed)
    mlm_probability: float = 0.15
    
    # Tokenization
    truncation: bool = True
    padding: bool = True
    
    # Memory optimization
    gradient_checkpointing: bool = False
    optim_target_modules: Optional[List[str]] = None

@dataclass
class GRPOConfig(TrainingConfig):
    """Group Relative Policy Optimization config"""
    
    # GRPO specific parameters
    group_size: int = 4
    beta: float = 0.1
    gamma: float = 0.99
    
    # Reward model
    reward_model_path: Optional[str] = None
    reward_key: str = "reward"
    
    # Data format
    group_key: str = "group"
    response_key: str = "response"