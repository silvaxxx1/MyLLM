# trainer/configs/classifier_config.py
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict, Any, Union
from enum import Enum
import logging
from .TrainerConfig import TrainerConfig, OptimizerType, SchedulerType, DeviceType

logger = logging.getLogger(__name__)

class PoolingStrategy(Enum):
    LAST = "last"
    MEAN = "mean" 
    MAX = "max"
    CLS = "cls"
    ATTENTION = "attention"

class ClassificationMetric(Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    AUC = "auc"

@dataclass
class ClassifierConfig(TrainerConfig):
    """
    Enhanced configuration for classification tasks with advanced features
    Built on top of the base TrainerConfig
    """
    
    # ----------------------------
    # Classification Specific
    # ----------------------------
    num_labels: int = field(default=2, metadata={"help": "Number of classification labels"})
    class_names: List[str] = field(default_factory=lambda: ["negative", "positive"], 
                                  metadata={"help": "Names of the classes for labeling"})
    pooling_strategy: PoolingStrategy = field(default=PoolingStrategy.LAST,
                                             metadata={"help": "How to pool sequence representations"})
    
    # ----------------------------
    # Classifier Head Architecture
    # ----------------------------
    classifier_dropout: float = field(default=0.1, 
                                     metadata={"help": "Dropout rate for classifier head"})
    classifier_hidden_size: Optional[int] = field(default=None,
                                                 metadata={"help": "Hidden size for classifier (None for linear)"})
    classifier_activation: str = field(default="gelu",
                                      metadata={"help": "Activation function for hidden layers"})
    classifier_num_layers: int = field(default=1,
                                      metadata={"help": "Number of layers in classifier head"})
    
    # ----------------------------
    # Differential Learning Rates
    # ----------------------------
    classifier_learning_rate: Optional[float] = field(default=None,
                                                     metadata={"help": "Separate LR for classifier head"})
    base_model_learning_rate: Optional[float] = field(default=None,
                                                     metadata={"help": "Separate LR for base model"})
    
    # ----------------------------
    # Classification Metrics & Evaluation
    # ----------------------------
    metric_for_best_model: str = field(default="eval_accuracy", 
                                     metadata={"help": "Primary metric for model selection"})
    greater_is_better: bool = field(default=True,
                                  metadata={"help": "Whether higher metric values are better"})
    additional_metrics: List[ClassificationMetric] = field(
        default_factory=lambda: [ClassificationMetric.F1, ClassificationMetric.PRECISION, ClassificationMetric.RECALL],
        metadata={"help": "Additional metrics to compute during evaluation"}
    )
    compute_confusion_matrix: bool = field(default=True,
                                         metadata={"help": "Whether to compute confusion matrix"})
    
    # ----------------------------
    # Data Augmentation & Regularization
    # ----------------------------
    class_weights: Optional[List[float]] = field(default=None,
                                               metadata={"help": "Manual class weights for imbalanced data"})
    label_smoothing: float = field(default=0.0,
                                  metadata={"help": "Label smoothing factor"})
    mixup_alpha: Optional[float] = field(default=None,
                                       metadata={"help": "Mixup augmentation alpha parameter"})
    
    # ----------------------------
    # Advanced Training
    # ----------------------------
    freeze_base_model: bool = field(default=False,
                                   metadata={"help": "Whether to freeze base model parameters"})
    unfreeze_layers: Optional[List[int]] = field(default=None,
                                               metadata={"help": "Specific layers to unfreeze if freezing"})
    gradual_unfreezing: bool = field(default=False,
                                   metadata={"help": "Whether to gradually unfreeze layers"})
    
    # ----------------------------
    # Enhanced Features (NEW)
    # ----------------------------
    gradient_checkpointing: bool = field(default=False,
                                       metadata={"help": "Enable gradient checkpointing for memory optimization"})
    use_mc_dropout: bool = field(default=False,
                                metadata={"help": "Enable Monte Carlo dropout for uncertainty estimation"})
    multi_task_config: Optional[Dict[str, int]] = field(default=None,
                                                      metadata={"help": "Configuration for multi-task learning"})
    confidence_tracking: bool = field(default=True,
                                    metadata={"help": "Track prediction confidence during training"})
    
    # ----------------------------
    # Advanced Pooling Configuration
    # ----------------------------
    attention_pooling_hidden_size: Optional[int] = field(default=None,
                                                       metadata={"help": "Hidden size for attention pooling"})
    
    # ----------------------------
    # Uncertainty Estimation
    # ----------------------------
    mc_dropout_samples: int = field(default=30,
                                   metadata={"help": "Number of samples for MC dropout"})
    confidence_threshold: float = field(default=0.7,
                                      metadata={"help": "Threshold for low confidence detection"})
    
    # ----------------------------
    # WandB Enhanced Logging
    # ----------------------------
    wandb_classification_plots: bool = field(default=True,
                                           metadata={"help": "Whether to log classification plots to WandB"})
    log_confidence_distribution: bool = field(default=True,
                                            metadata={"help": "Log confidence distribution to WandB"})
    log_uncertainty_metrics: bool = field(default=True,
                                        metadata={"help": "Log uncertainty metrics to WandB"})
    
    def __post_init__(self):
        """Extended validation for classification-specific parameters"""
        # First call parent validation
        super().__post_init__()
        
        logger.info("Validating Enhanced ClassifierConfig...")
        
        # Convert string pooling strategy to enum if needed
        if isinstance(self.pooling_strategy, str):
            try:
                self.pooling_strategy = PoolingStrategy(self.pooling_strategy.lower())
                logger.info(f"Converted pooling_strategy from string to enum: {self.pooling_strategy}")
            except ValueError:
                raise ValueError(f"Invalid pooling_strategy: {self.pooling_strategy}. Must be one of: {[e.value for e in PoolingStrategy]}")
        
        # Convert string additional_metrics to enums if needed
        if self.additional_metrics and isinstance(self.additional_metrics[0], str):
            converted_metrics = []
            for metric in self.additional_metrics:
                try:
                    converted_metrics.append(ClassificationMetric(metric.lower()))
                except ValueError:
                    logger.warning(f"Invalid metric: {metric}. Skipping.")
            self.additional_metrics = converted_metrics
        
        # Classification-specific validation
        if self.num_labels < 2:
            raise ValueError("num_labels must be at least 2 for classification")
        
        if len(self.class_names) != self.num_labels:
            raise ValueError(f"class_names length ({len(self.class_names)}) must match num_labels ({self.num_labels})")
        
        if self.classifier_dropout < 0 or self.classifier_dropout >= 1:
            raise ValueError("classifier_dropout must be between 0 and 1")
        
        if self.classifier_hidden_size is not None and self.classifier_hidden_size <= 0:
            raise ValueError("classifier_hidden_size must be positive if specified")
        
        if self.label_smoothing < 0 or self.label_smoothing >= 1:
            raise ValueError("label_smoothing must be between 0 and 1")
        
        if self.class_weights and len(self.class_weights) != self.num_labels:
            raise ValueError(f"class_weights length ({len(self.class_weights)}) must match num_labels ({self.num_labels})")
        
        # Enhanced feature validation
        if self.multi_task_config:
            if not isinstance(self.multi_task_config, dict):
                raise ValueError("multi_task_config must be a dictionary")
            for task_name, num_labels in self.multi_task_config.items():
                if not isinstance(task_name, str):
                    raise ValueError("multi_task_config keys must be strings")
                if not isinstance(num_labels, int) or num_labels < 2:
                    raise ValueError("multi_task_config values must be integers >= 2")
        
        if self.mc_dropout_samples < 1:
            raise ValueError("mc_dropout_samples must be at least 1")
        
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        # Set default learning rates if not specified
        if self.classifier_learning_rate is None:
            self.classifier_learning_rate = self.learning_rate * 10
            logger.info(f"Using default classifier_learning_rate: {self.classifier_learning_rate}")
        
        if self.base_model_learning_rate is None:
            self.base_model_learning_rate = self.learning_rate
            logger.info(f"Using base_model_learning_rate: {self.base_model_learning_rate}")
        
        # Set attention pooling hidden size if using attention pooling
        if (self.pooling_strategy == PoolingStrategy.ATTENTION and 
            self.attention_pooling_hidden_size is None):
            # Default to half of model hidden size (will be set in trainer)
            self.attention_pooling_hidden_size = 512  # Reasonable default
            logger.info(f"Using default attention_pooling_hidden_size: {self.attention_pooling_hidden_size}")
        
        # Ensure metric_for_best_model makes sense for classification
        classification_metrics = ["accuracy", "f1", "precision", "recall", "auc"]
        if not any(metric in self.metric_for_best_model for metric in classification_metrics):
            logger.warning(f"metric_for_best_model '{self.metric_for_best_model}' may not be optimal for classification. Consider using: {classification_metrics}")
        
        # FIXED: Handle both enum and string cases for logging
        pooling_value = self.pooling_strategy.value if hasattr(self.pooling_strategy, 'value') else str(self.pooling_strategy)
        
        # Log enhanced features status
        enhanced_features = []
        if self.gradient_checkpointing:
            enhanced_features.append("gradient_checkpointing")
        if self.use_mc_dropout:
            enhanced_features.append("MC_dropout")
        if self.multi_task_config:
            enhanced_features.append(f"multi_task({len(self.multi_task_config)} tasks)")
        if self.pooling_strategy == PoolingStrategy.ATTENTION:
            enhanced_features.append("attention_pooling")
        
        logger.info(f"Enhanced ClassifierConfig validation complete: {self.num_labels} classes, pooling={pooling_value}")
        if enhanced_features:
            logger.info(f"Enabled enhanced features: {', '.join(enhanced_features)}")
    
    def get_classifier_config_dict(self) -> Dict[str, Any]:
        """Get classifier-specific configuration for logging"""
        base_config = self.to_dict()
        
        # Add classifier-specific config
        classifier_config = {
            "num_labels": self.num_labels,
            "class_names": self.class_names,
            "pooling_strategy": self.pooling_strategy.value if hasattr(self.pooling_strategy, 'value') else self.pooling_strategy,
            "classifier_dropout": self.classifier_dropout,
            "classifier_hidden_size": self.classifier_hidden_size,
            "classifier_learning_rate": self.classifier_learning_rate,
            "base_model_learning_rate": self.base_model_learning_rate,
            "freeze_base_model": self.freeze_base_model,
            # Enhanced features
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_mc_dropout": self.use_mc_dropout,
            "multi_task_config": self.multi_task_config,
            "confidence_tracking": self.confidence_tracking,
            "attention_pooling_hidden_size": self.attention_pooling_hidden_size,
        }
        
        return {**base_config, **classifier_config}
    
    def get_optimizer_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups for differential learning rates"""
        groups = []
        
        # Base model parameters
        base_lr = self.base_model_learning_rate or self.learning_rate
        groups.append({
            "params": [],  # Will be filled by trainer
            "lr": base_lr,
            "name": "base_model"
        })
        
        # Classifier parameters (usually higher LR)
        classifier_lr = self.classifier_learning_rate or (base_lr * 10)
        groups.append({
            "params": [],  # Will be filled by trainer  
            "lr": classifier_lr,
            "name": "classifier"
        })
        
        # Additional groups for enhanced features
        if self.pooling_strategy == PoolingStrategy.ATTENTION:
            groups.append({
                "params": [],  # Will be filled by trainer
                "lr": classifier_lr,  # Same as classifier
                "name": "attention_pooler"
            })
        
        if self.multi_task_config:
            groups.append({
                "params": [],  # Will be filled by trainer
                "lr": classifier_lr,  # Same as classifier
                "name": "multi_task"
            })
        
        return groups
    
    def get_enhanced_features_summary(self) -> Dict[str, Any]:
        """Get summary of enabled enhanced features"""
        return {
            "gradient_checkpointing": self.gradient_checkpointing,
            "mc_dropout": self.use_mc_dropout,
            "multi_task_learning": bool(self.multi_task_config),
            "attention_pooling": self.pooling_strategy == PoolingStrategy.ATTENTION,
            "confidence_tracking": self.confidence_tracking,
            "class_imbalance_handling": bool(self.class_weights),
            "num_enhanced_features": sum([
                self.gradient_checkpointing,
                self.use_mc_dropout,
                bool(self.multi_task_config),
                self.pooling_strategy == PoolingStrategy.ATTENTION,
                self.confidence_tracking,
                bool(self.class_weights)
            ])
        }

# ----------------------------
# Pre-configured Configs for Common Use Cases
# ----------------------------

@dataclass  
class TextClassificationConfig(ClassifierConfig):
    """Pre-configured for general text classification"""
    def __post_init__(self):
        self.pooling_strategy = PoolingStrategy.MEAN
        self.classifier_hidden_size = 512
        self.classifier_dropout = 0.2
        super().__post_init__()

@dataclass
class SentimentAnalysisConfig(ClassifierConfig):
    """Pre-configured for sentiment analysis"""
    def __post_init__(self):
        self.num_labels = 3
        self.class_names = ["negative", "neutral", "positive"]
        self.pooling_strategy = PoolingStrategy.LAST
        self.metric_for_best_model = "eval_f1"
        super().__post_init__()

@dataclass
class IntentClassificationConfig(ClassifierConfig):
    """Pre-configured for intent classification"""
    def __post_init__(self):
        self.pooling_strategy = PoolingStrategy.CLS
        self.classifier_num_layers = 2
        self.metric_for_best_model = "eval_accuracy"
        super().__post_init__()

@dataclass
class FewShotClassifierConfig(ClassifierConfig):
    """Pre-configured for few-shot learning scenarios"""
    def __post_init__(self):
        self.freeze_base_model = True
        self.classifier_learning_rate = 1e-3
        self.num_epochs = 10  # Usually need more epochs for few-shot
        self.batch_size = 4   # Smaller batches for few-shot
        super().__post_init__()

@dataclass
class EnhancedClassifierConfig(ClassifierConfig):
    """Pre-configured with all enhanced features enabled"""
    def __post_init__(self):
        # Enable all enhanced features
        self.gradient_checkpointing = True
        self.use_mc_dropout = True
        self.confidence_tracking = True
        self.pooling_strategy = PoolingStrategy.ATTENTION
        self.log_confidence_distribution = True
        self.log_uncertainty_metrics = True
        
        # Multi-task configuration (can be overridden)
        if not self.multi_task_config:
            self.multi_task_config = {
                "main": self.num_labels,
                "auxiliary": 2  # Default auxiliary task
            }
        
        super().__post_init__()

@dataclass
class ProductionClassifierConfig(ClassifierConfig):
    """Pre-configured for production deployment with optimal settings"""
    def __post_init__(self):
        self.gradient_checkpointing = True  # Memory efficiency
        self.confidence_tracking = True     # Monitor predictions
        self.pooling_strategy = PoolingStrategy.MEAN  # Robust pooling
        self.classifier_dropout = 0.2       # Better regularization
        self.label_smoothing = 0.1          # Better calibration
        self.compute_confusion_matrix = True # Detailed evaluation
        super().__post_init__()