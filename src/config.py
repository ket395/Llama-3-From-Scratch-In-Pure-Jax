from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ModelArgs:
    # Model architecture
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = 2
    vocab_size: int = 50257  # GPT-2 vocabulary size
    multiple_of: int = 32
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 512
    dropout_rate: float = 0.0

@dataclass(frozen=True)
class TrainingArgs:
    # Training hyperparameters
    batch_size: int = 16
    base_learning_rate: float = 3e-4
    num_epochs: int = 30
    steps_per_epoch: int = 1000
    warmup_steps: int = 100
    
    # Adam optimizer parameters
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8 