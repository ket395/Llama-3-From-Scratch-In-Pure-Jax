import jax
import jax.numpy as jnp
from jax import random
import tiktoken
import numpy as np
from functools import partial
import time
import os
from jax.tree_util import tree_leaves
import pickle
import wandb
from config import ModelArgs, TrainingArgs
from model import init_params, model_forward
from inference import generate, load_model, save_model

os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

print("JAX devices:", jax.devices())

def get_sequence(data, idx, block_size):
    x = data[idx:idx+block_size]
    y = data[idx+1:idx+block_size+1]
    return x, y

def get_batch(data, batch_size, block_size):
    ix = np.random.randint(0, len(data) - block_size, batch_size)
    vectorized_get_sequence = jax.vmap(partial(get_sequence, data), in_axes=(0, None))
    return vectorized_get_sequence(ix, block_size)

def single_example_loss(logit, target, vocab_size):
    logit = logit.reshape(-1, vocab_size)
    target = target.reshape(-1)
    labels = jax.nn.one_hot(target, vocab_size)
    return -jnp.sum(labels * jax.nn.log_softmax(logit, axis=-1))

@partial(jax.jit, static_argnums=(2,))
def compute_loss(params, batch, args):
    inputs, targets = batch
    logits, _ = model_forward(params, inputs, args)
    vectorized_loss = jax.vmap(partial(single_example_loss, vocab_size=args.vocab_size))
    losses = vectorized_loss(logits, targets)
    return jnp.mean(losses)

def update_momentum(m, g, beta1):
    return beta1 * m + (1 - beta1) * g

def update_velocity(v, g, beta2):
    return beta2 * v + (1 - beta2) * jnp.square(g)

def compute_m_hat(m, beta1, t):
    return m / (1 - beta1 ** t)

def compute_v_hat(v, beta2, t):
    return v / (1 - beta2 ** t)

def update_param(p, m_hat, v_hat, learning_rate, eps):
    return p - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

@partial(jax.jit, static_argnums=(3,))
def update_step(params, adam_state, batch, args, learning_rate, beta1, beta2, eps):
    loss, grads = jax.value_and_grad(compute_loss)(params, batch, args)
    t = adam_state['t'] + 1
    
    vectorized_momentum = jax.vmap(partial(update_momentum, beta1=beta1))
    vectorized_velocity = jax.vmap(partial(update_velocity, beta2=beta2))
    
    m = jax.tree.map(vectorized_momentum, adam_state['m'], grads)
    v = jax.tree.map(vectorized_velocity, adam_state['v'], grads)
    
    vectorized_m_hat = jax.vmap(partial(compute_m_hat, beta1=beta1, t=t))
    vectorized_v_hat = jax.vmap(partial(compute_v_hat, beta2=beta2, t=t))
    
    m_hat = jax.tree.map(vectorized_m_hat, m)
    v_hat = jax.tree.map(vectorized_v_hat, v)
    
    vectorized_update = jax.vmap(partial(update_param, learning_rate=learning_rate, eps=eps))
    new_params = jax.tree.map(vectorized_update, params, m_hat, v_hat)
    
    new_adam_state = {
        'm': m,
        'v': v,
        't': t
    }
    return new_params, new_adam_state, loss

def get_lr(step, base_lr, warmup_steps=100, total_steps=5000):
    warmup_factor = jnp.minimum(1.0, step / warmup_steps)
    progress = jnp.maximum(0.0, (step - warmup_steps) / (total_steps - warmup_steps))
    decay_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    return base_lr * warmup_factor * decay_factor

# Initialize tokenizer and load data
enc = tiktoken.get_encoding("gpt2")
with open('shakespeare.txt', 'r') as f:
    text = f.read()
tokens = enc.encode(text)
data = np.array(tokens, dtype=np.int32)

# Model initialization
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

batch_size = 16
base_learning_rate = 3e-4
num_epochs = 2
steps_per_epoch = 1000
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

key = random.PRNGKey(0)
params = init_params(key, args)
adam_state = {
    'm': jax.tree.map(jnp.zeros_like, params),
    'v': jax.tree.map(jnp.zeros_like, params),
    't': 0
}

# Initialize wandb
wandb.init(project="your_project_name", config={
    "learning_rate": base_learning_rate,
    "epochs": num_epochs,
    "batch_size": batch_size,
    "model_dim": args.dim,
    "n_layers": args.n_layers,
    "n_heads": args.n_heads,
})

# Training loop
print("Starting training...")
print(f"Training for {num_epochs} epochs, {steps_per_epoch} steps per epoch")
print(f"Total steps: {num_epochs * steps_per_epoch}")
total_params = sum(x.size for x in tree_leaves(params))
print(f"Total parameters: {total_params:,}")

global_step = 0
for epoch in range(num_epochs):
    epoch_loss = 0.0
    print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
    
    for i in range(steps_per_epoch):
        X, Y = get_batch(data, batch_size, args.max_seq_len)
        batch = (jnp.array(X), jnp.array(Y))
        lr = get_lr(global_step, base_learning_rate)
        params, adam_state, loss = update_step(
            params, adam_state, batch, args, 
            lr, beta1, beta2, eps
        )
        epoch_loss += loss
        wandb.log({"epoch": epoch + 1, "step": global_step, "loss": loss, "learning_rate": lr})
        
        if i % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Step {i}, Loss: {loss:.4f}, LR: {lr:.6f}")
        
        global_step += 1
    
    avg_epoch_loss = epoch_loss / steps_per_epoch
    print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

print("\ngreat job ")
print("\nsaving model weights waitttt...")
save_model(params, 'model_weights.pkl')
