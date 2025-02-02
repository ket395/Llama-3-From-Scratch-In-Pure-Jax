from dataclasses import dataclass
import tiktoken

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

@dataclass
class ModelArgs:
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    max_seq_len: int
    multiple_of: int = 32
    norm_eps: float = 1e-5
    dropout_rate: float = 0.0

# Model configuration
args = ModelArgs(
    vocab_size=enc.n_vocab,
    dim=768,           
    n_layers=12,       
    n_heads=12,        
    n_kv_heads=2,      
    max_seq_len=512,   
    multiple_of=32,    
    norm_eps=1e-5,     
)

# Training parameters
batch_size = 16
base_learning_rate = 3e-4
num_epochs = 30  
steps_per_epoch = 1000  

# Adam hyperparameters
beta1 = 0.9
beta2 = 0.999
eps = 1e-8