#imports 

import jax
import jax.numpy as jnp
from jax import random
import tiktoken
from model import ModelArgs, model_forward
import os
import pickle
import time 

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

args = ModelArgs(
    vocab_size=50257,  # GPT-2 vocabulary size
    dim=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=2,
    max_seq_len=512,
    multiple_of=32,
    norm_eps=1e-5,
)

# Force JAX to use GPU lol 
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# After imports, verify GPU is available
print("JAX devices:", jax.devices())




def load_model(filename):

    with open(filename, 'rb') as f:
        return pickle.load(f)

def top_k_top_p_filtering(logits, top_k=50, top_p=0.9, temperature=0.7):
   
    if len(logits.shape) == 1:
        logits = logits[None, :]
    
    vocab_size = logits.shape[-1]
    logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, vocab_size)
        top_k_logits = jax.lax.top_k(logits, top_k)[0]
        indices_to_remove = logits < top_k_logits[..., -1:]
        logits = jnp.where(indices_to_remove, float('-inf'), logits)
    
    # Top-p filtering
    if top_p > 0.0:
        sorted_logits = jnp.sort(logits, axis=-1)[:, ::-1]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
        sort_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)
        
        # Create a boolean mask of the same shape as logits
        indices_to_remove = jnp.zeros_like(logits, dtype=bool)
        indices_to_remove = indices_to_remove.at[jnp.arange(logits.shape[0])[:, None], sort_indices].set(sorted_indices_to_remove)
        
        logits = jnp.where(indices_to_remove, jnp.array(-1e10), logits)
    
    return logits[0]  

def generate(params, prompt_tokens, max_new_tokens, args, temperature=0.7, top_k=50, top_p=0.9):
  
    key = random.PRNGKey(int(time.time() * 1000))
    x = jnp.array(prompt_tokens)[None, :]  # [1, seq_len]
    
    for _ in range(max_new_tokens):
        # Get logits and ignore cache
        logits, _ = model_forward(params, x, args)
        
        # Get logits of last token and apply temperature
        logits = logits[0, -1, :]
        
        # Apply top-k and top-p filtering
        filtered_logits = top_k_top_p_filtering(logits, top_k, top_p, temperature)
        
        # Sample from the distribution
        key, subkey = random.split(key)
        next_token = random.categorical(subkey, filtered_logits)
        
        # Append to the sequence
        x = jnp.concatenate([x, next_token[None, None]], axis=1)
    
    return x[0]

# Load saved weights
params = load_model('model_weights.pkl')

# Generate text
prompt = "Hi how are"
prompt_tokens = enc.encode(prompt)
generated_tokens = generate(params, prompt_tokens, 50, args, temperature=0.7, top_k=50, top_p=0.9)
generated_text = enc.decode(generated_tokens.tolist())
print("\nGenerated text:")
print(generated_text)
