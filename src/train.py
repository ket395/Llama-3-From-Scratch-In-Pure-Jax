import jax
import jax.numpy as jnp
from jax import random
import tiktoken
import numpy as np
from functools import partial
import time
import os
from jax.tree_util import tree_map, tree_leaves
import wandb
from dataclasses import dataclass

# Import the new initialization functions
from model import init_model_params, model_forward
from utils import get_batch, get_lr, save_model, load_model, generate
from config import ModelArgs, args, batch_size, base_learning_rate, num_epochs, steps_per_epoch, beta1, beta2, eps, enc

os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

print("JAX devices:", jax.devices())

# Load and prepare data
enc = tiktoken.get_encoding("gpt2")
with open('shakespeare.txt', 'r') as f:
    text = f.read()
tokens = enc.encode(text)
data = np.array(tokens, dtype=np.int32)

# Initialize model parameters with new structure
key = random.PRNGKey(0)
params = init_model_params(
    key=key,
    vocab_size=args.vocab_size,
    dim=args.dim,
    n_layers=args.n_layers,
    n_heads=args.n_heads,
    n_kv_heads=args.n_kv_heads
)

# Rest of the initialization code remains the same
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

# Training loop
print(f"Total parameters: {sum(x.size for x in tree_leaves(params)):,}")
print(f"Training for {num_epochs} epochs, {steps_per_epoch} steps per epoch")

total_steps = num_epochs * steps_per_epoch

for step in range(total_steps):
    current_epoch = step // steps_per_epoch
    batch = tuple(jnp.array(x) for x in get_batch(data, batch_size, args.max_seq_len))
    lr = get_lr(step, base_learning_rate)
    
    params, adam_state, loss = update_step(
        params, adam_state, batch, 
        args.vocab_size, args.dim, args.n_layers, args.n_heads, 
        args.n_kv_heads, args.max_seq_len,
        lr, beta1, beta2, eps
    )
    
    wandb.log({"loss": loss, "learning_rate": lr})
    
    if step % 50 == 0:
        print(f"Step {step}/{total_steps} | Epoch {current_epoch+1}/{num_epochs} | Loss: {loss:.4f} | LR: {lr:.6f}")
    
    if (step + 1) % steps_per_epoch == 0:
        print(f"Epoch {current_epoch+1} completed | Loss: {loss:.4f}")

# Save model and generate sample
save_model(params, 'model_weights.pkl')

prompt = "O, fair maiden, thy beauty doth outshine the stars."
prompt_tokens = enc.encode(prompt)
generated_tokens = generate(params, prompt_tokens, 100, args, model_forward, temperature=0.8)
print("\nSample generation:", enc.decode(generated_tokens.tolist()))