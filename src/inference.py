# To load and use the model later:
import jax
import jax.numpy as jnp
from jax import random
import tiktoken
import numpy as np
from model import ModelArgs, init_params, model_forward
from functools import partial
import time
import os
from jax.tree_util import tree_map, tree_leaves
from jax.experimental import checkify
import pickle

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

args = ModelArgs(
    vocab_size=enc.n_vocab,
    dim=768,           # increased dimension for more parameters
    n_layers=12,       # increased layers
    n_heads=12,        # increased heads
    n_kv_heads=2,      # kept kv heads
    max_seq_len=512,   # kept same sequence length
    multiple_of=32,    # kept same
    norm_eps=1e-5,     # kept same
)

# Force JAX to use GPU
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# After imports, verify GPU is available
print("JAX devices:", jax.devices())




def load_model(filename):
    """Load model parameters from a file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
    """
    Applies repetition penalty to already generated tokens.
    penalty > 1.0 means we penalize repetitions
    """
    # Convert to numpy for indexing
    logits_np = np.array(logits)
    
    # Get unique tokens that have been generated
    unique_tokens = set(generated_tokens)
    
    # Apply penalty to all previously generated tokens
    for token in unique_tokens:
        logits_np[token] = logits_np[token] / penalty
    
    return jnp.array(logits_np)

def apply_length_penalty(logits, current_length, max_length, alpha=0.7):
    """
    Applies length penalty to encourage or discourage longer sequences
    alpha < 1.0 encourages shorter sequences
    alpha > 1.0 encourages longer sequences
    """
    length_ratio = current_length / max_length
    penalty = (5 + length_ratio) / 6
    return logits / (penalty ** alpha)

def top_p_sampling(logits, p, temperature=1.0, prng_key=None):
    """
    Implements nucleus (top-p) sampling on logits.
    """
    # Apply temperature
    logits = logits / temperature
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = jax.lax.top_k(logits, logits.shape[-1])
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to keep also the first token above the threshold
    sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1)
    sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
    
    # Create the new logits with only top-p tokens
    sorted_logits = jnp.where(sorted_indices_to_remove, -jnp.inf, sorted_logits)
    
    # Sample from the filtered distribution
    probs = jax.nn.softmax(sorted_logits)
    selected_idx = random.categorical(prng_key, sorted_logits[None, :])[0]
    
    return sorted_indices[selected_idx]

def advanced_sampling(logits, generated_tokens, current_length, max_length, 
                     temperature=0.7, top_k=40, top_p=0.9, 
                     repetition_penalty=1.2, length_penalty_alpha=0.7,
                     prng_key=None):
    """
    Combines multiple sampling strategies for better text generation
    """
    # Apply repetition penalty
    logits = apply_repetition_penalty(logits, generated_tokens, repetition_penalty)
    
    # Apply length penalty
    logits = apply_length_penalty(logits, current_length, max_length, length_penalty_alpha)
    
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-k filtering
    top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
    
    # Apply top-p filtering
    sorted_logits, sorted_indices = jax.lax.top_k(top_k_logits, top_k)
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1)
    sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
    
    # Create final logits
    final_logits = jnp.where(sorted_indices_to_remove, -jnp.inf, sorted_logits)
    
    # Sample from the filtered distribution
    probs = jax.nn.softmax(final_logits)
    selected_idx = random.categorical(prng_key, final_logits[None, :])[0]
    
    return top_k_indices[sorted_indices[selected_idx]]

def generate(params, prompt_tokens, max_new_tokens, args, 
            temperature=0.7, top_k=40, top_p=0.9,
            repetition_penalty=1.2, length_penalty_alpha=0.7):
    """Enhanced generation function with multiple sampling strategies"""
    key = random.PRNGKey(int(time.time() * 1000))
    x = jnp.array(prompt_tokens)[None, :]
    generated_tokens = list(prompt_tokens)
    
    # Initialize cache
    cache = None
    
    for i in range(max_new_tokens):
        # Get logits and update cache
        logits, cache = model_forward(params, x, args, cache)
        logits = logits[0, -1, :]
        
        # Generate next token using advanced sampling
        key, subkey = random.split(key)
        next_token = advanced_sampling(
            logits,
            generated_tokens,
            len(generated_tokens),
            len(prompt_tokens) + max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty_alpha=length_penalty_alpha,
            prng_key=subkey
        )
        
        # For the next iteration, we only need to process the last token
        x = next_token[None, None]
        generated_tokens.append(next_token.item())
        
        # Early stopping conditions
        if next_token.item() == enc.eot_token:
            break
        
        # Optional: Add some basic safety checks
        if len(generated_tokens) > args.max_seq_len:
            break
    
    return jnp.array(generated_tokens)

def view_architecture_from_weights(params):
    """
    Traverses the model weights dictionary and prints the architecture in terms of
    module names and parameter shapes. Note: This only shows the dictionary structure
    of saved parameters, not a full instance of the model.
    """
    def print_dict(dic, indent=0):
        for key, value in dic.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_dict(value, indent + 1)
            elif hasattr(value, "shape"):
                print(f"{prefix}{key}: shape {value.shape}")
            else:
                print(f"{prefix}{key}: {value}")
                
    print("Model Architecture (from weights):")
    print("=" * 50)
    print_dict(params)
    print("=" * 50)

# Load saved weights
params = load_model('model_weights.pkl')
view_architecture_from_weights(params)

# Generate text
prompt = "Hi how are you "
prompt_tokens = enc.encode(prompt)
generated_tokens = generate(
    params,
    prompt_tokens,
    max_new_tokens=50,
    args=args,
    temperature=0.8,          # Controls randomness
    top_k=40,                # Limits to top 40 tokens
    top_p=0.9,               # Nucleus sampling threshold
    repetition_penalty=1.2,   # Penalize repetitions
    length_penalty_alpha=0.7  # Prefer slightly shorter sequences
)
generated_text = enc.decode(generated_tokens.tolist())
print("\nGenerated text:")
print(generated_text)
