import jax
import jax.numpy as jnp
from jax import random
import tiktoken
import numpy as np
from model import  init_params, model_forward
from functools import partial
import time
import os
from jax.tree_util import tree_map, tree_leaves
from jax.experimental import checkify
import pickle
import wandb
from dataclasses import dataclass
from utils import get_batch, get_lr, save_model, load_model, generate
from config import ModelArgs, args, batch_size, base_learning_rate, num_epochs, steps_per_epoch, beta1, beta2, eps, enc

os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

print("JAX devices:", jax.devices())

enc = tiktoken.get_encoding("gpt2")

with open('shakespeare.txt', 'r') as f:
    text = f.read()
tokens = enc.encode(text)
data = np.array(tokens, dtype=np.int32)

key = random.PRNGKey(0)
params = init_params(key, args)

adam_state = {
    'm': tree_map(jnp.zeros_like, params),
    'v': tree_map(jnp.zeros_like, params),
    't': 0
}

wandb.init(project="your_project_name", config={
    "learning_rate": base_learning_rate,
    "epochs": num_epochs,
    "batch_size": batch_size,
    "model_dim": args.dim,
    "n_layers": args.n_layers,
    "n_heads": args.n_heads,
})

@partial(jax.checkpoint, static_argnums=(2,3,4,5,6,7))
def forward_fn(params, inputs, vocab_size, dim, n_layers, n_heads, n_kv_heads, max_seq_len):
    args_dict = {
        'vocab_size': vocab_size,
        'dim': dim,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'max_seq_len': max_seq_len,
        'multiple_of': 32,
        'norm_eps': 1e-5,
    }
    return model_forward(params, inputs, ModelArgs(**args_dict))


# First, define the loss function at module level
def compute_loss(params, batch, vocab_size, dim, n_layers, n_heads, n_kv_heads, max_seq_len):
    inputs, targets = batch
    logits, _ = forward_fn(params, inputs, vocab_size, dim, n_layers, n_heads, n_kv_heads, max_seq_len)
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)
    labels = jax.nn.one_hot(targets, vocab_size)
    loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1))
    
    return loss / targets.size



@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def update_step(params, adam_state, batch, vocab_size, dim, n_layers, n_heads, n_kv_heads, max_seq_len, learning_rate, beta1, beta2, eps):
    loss, grads = jax.value_and_grad(lambda p, b: compute_loss(p, b, vocab_size, dim, n_layers, n_heads, n_kv_heads, max_seq_len))(params, batch)
    
    t = adam_state['t'] + 1
    m = tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, adam_state['m'], grads)
    v = tree_map(lambda v, g: beta2 * v + (1 - beta2) * jnp.square(g), adam_state['v'], grads)
    m_hat = tree_map(lambda m: m / (1 - beta1 ** t), m)
    v_hat = tree_map(lambda v: v / (1 - beta2 ** t), v)
    new_params = tree_map(
        lambda p, m_hat, v_hat: p - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps),
        params, m_hat, v_hat
    )
    new_adam_state = {'m': m, 'v': v, 't': t}
    return new_params, new_adam_state, loss




print(f"Total parameters: {sum(x.size for x in tree_leaves(params)):,}")
print(f"Training for {num_epochs} epochs, {steps_per_epoch} steps per epoch")


# Calculate total number of steps
total_steps = num_epochs * steps_per_epoch

# Flatten the loops into a single loop
for step in range(total_steps):
    # Calculate current epoch for logging
    current_epoch = step // steps_per_epoch
    
    # Get batch
    batch = tuple(jnp.array(x) for x in get_batch(data, batch_size, args.max_seq_len))
    
    # Calculate learning rate
    lr = get_lr(step, base_learning_rate)
    
    # Update step
    params, adam_state, loss = update_step(
        params, adam_state, batch, 
        args.vocab_size, args.dim, args.n_layers, args.n_heads, 
        args.n_kv_heads, args.max_seq_len,
        lr, beta1, beta2, eps
    )
    
    # Log to wandb
    wandb.log({"loss": loss, "learning_rate": lr})
    
    # Print progress
    if step % 50 == 0:
        print(f"Step {step}/{total_steps} | Epoch {current_epoch+1}/{num_epochs} | Loss: {loss:.4f} | LR: {lr:.6f}")
    
    # Calculate and print epoch statistics
    if (step + 1) % steps_per_epoch == 0:
        print(f"Epoch {current_epoch+1} completed | Loss: {loss:.4f}")
        
        

save_model(params, 'model_weights.pkl')

prompt = "O, fair maiden, thy beauty doth outshine the stars."
prompt_tokens = enc.encode(prompt)
generated_tokens = generate(params, prompt_tokens, 100, args, model_forward, temperature=0.8)
print("\nSample generation:", enc.decode(generated_tokens.tolist()))