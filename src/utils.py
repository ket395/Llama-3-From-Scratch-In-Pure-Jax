import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pickle
import time

def get_batch(data, batch_size, block_size):
    ix = np.random.randint(0, len(data) - block_size, batch_size)
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def get_lr(step, base_lr, warmup_steps=100, total_steps=5000):
    warmup_factor = jnp.minimum(1.0, step / warmup_steps)
    progress = jnp.maximum(0.0, (step - warmup_steps) / (total_steps - warmup_steps))
    decay_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    return base_lr * warmup_factor * decay_factor

def save_model(params, filename):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def generate(params, prompt_tokens, max_new_tokens, args, model_forward, temperature=0.7):
    key = random.PRNGKey(int(time.time() * 1000))
    x = jnp.array(prompt_tokens)[None, :]  # [1, seq_len]
    
    for _ in range(max_new_tokens):
        logits, _ = model_forward(params, x, args)
        logits = logits[0, -1, :] / temperature
        key, subkey = random.split(key)
        next_token = random.categorical(subkey, logits[None, :])[0]
        x = jnp.concatenate([x, next_token[None, None]], axis=1)
    
    return x[0]