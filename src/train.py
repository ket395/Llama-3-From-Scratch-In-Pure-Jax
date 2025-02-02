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
import wandb

# Force JAX to use GPU
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# After imports, verify GPU is available
print("JAX devices:", jax.devices())

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load and tokenize data
with open('shakespeare.txt', 'r') as f:
    text = f.read()
tokens = enc.encode(text)
data = np.array(tokens, dtype=np.int32)

# Model hyperparameters - adjusted for ~125M parameters
# Total params ≈ dim * (vocab_size + dim + n_layers * 12 * dim)
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

# Training parameters
batch_size = 16
base_learning_rate = 3e-4
num_epochs = 30  # Increased number of epochs (can be adjusted to 35 as needed)
steps_per_epoch = 1000  # Made explicit

# Adam hyperparameters
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

# Initialize model
key = random.PRNGKey(0)
params = init_params(key, args)

# Initialize Adam state
adam_state = {
    'm': tree_map(jnp.zeros_like, params),  # First moment
    'v': tree_map(jnp.zeros_like, params),  # Second moment
    't': 0  # Timestep
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

def get_batch(data, batch_size, block_size):
    ix = np.random.randint(0, len(data) - block_size, batch_size)
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Create forward function with static dimensions
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

@partial(jax.jit, static_argnums=(2,))
def compute_loss(params, batch, args):
    inputs, targets = batch  # Unpack the tuple directly
    
    # Get logits from model (and ignore the cache)
    logits, _ = model_forward(params, inputs, args)
    
    # Now we can reshape the logits
    logits = logits.reshape(-1, args.vocab_size)
    targets = targets.reshape(-1)
    labels = jax.nn.one_hot(targets, args.vocab_size)
    
    loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1))
    return loss / targets.size

def get_lr(step, base_lr, warmup_steps=100, total_steps=5000):
    """Learning rate schedule with warmup and cosine decay"""
    # Linear warmup
    warmup_factor = jnp.minimum(1.0, step / warmup_steps)
    
    # Cosine decay
    progress = jnp.maximum(0.0, (step - warmup_steps) / (total_steps - warmup_steps))
    decay_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    
    return base_lr * warmup_factor * decay_factor

@partial(jax.jit, static_argnums=(3,))
def update_step(params, adam_state, batch, args, learning_rate, beta1, beta2, eps):
    """Adam optimizer update step"""
    loss, grads = jax.value_and_grad(compute_loss)(params, batch, args)
    
    # Update timestep
    t = adam_state['t'] + 1
    
    # Update biased first moment estimate
    m = tree_map(
        lambda m, g: beta1 * m + (1 - beta1) * g,
        adam_state['m'],
        grads
    )
    
    # Update biased second raw moment estimate
    v = tree_map(
        lambda v, g: beta2 * v + (1 - beta2) * jnp.square(g),
        adam_state['v'],
        grads
    )
    
    # Compute bias-corrected first moment estimate
    m_hat = tree_map(lambda m: m / (1 - beta1 ** t), m)
    
    # Compute bias-corrected second raw moment estimate
    v_hat = tree_map(lambda v: v / (1 - beta2 ** t), v)
    
    # Update parameters
    new_params = tree_map(
        lambda p, m_hat, v_hat: p - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps),
        params,
        m_hat,
        v_hat
    )
    
    # Create new state
    new_adam_state = {
        'm': m,
        'v': v,
        't': t
    }
    
    return new_params, new_adam_state, loss

# Training loop
print("Starting training...")
print(f"Training for {num_epochs} epochs, {steps_per_epoch} steps per epoch")
print(f"Total steps: {num_epochs * steps_per_epoch}")

# Print parameter count
total_params = sum(x.size for x in tree_leaves(params))
print(f"Total parameters: {total_params:,}")

global_step = 0
for epoch in range(num_epochs):
    epoch_loss = 0.0
    print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
    
    for i in range(steps_per_epoch):
        # Get batch
        X, Y = get_batch(data, batch_size, args.max_seq_len)
        batch = (jnp.array(X), jnp.array(Y))
        
        # Calculate learning rate
        lr = get_lr(global_step, base_learning_rate)
        
        # Update parameters using Adam
        params, adam_state, loss = update_step(
            params, adam_state, batch, args, 
            lr, beta1, beta2, eps
        )
        
        epoch_loss += loss
        
        # Log loss to wandb
        wandb.log({"epoch": epoch + 1, "step": global_step, "loss": loss, "learning_rate": lr})
        
        if i % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Step {i}, Loss: {loss:.4f}, LR: {lr:.6f}")
        
        global_step += 1
    
    # Print epoch summary
    avg_epoch_loss = epoch_loss / steps_per_epoch
    print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

print("\nTraining completed!")

# Save model weights after training
def save_model(params, filename):
    """Save model parameters to a file"""
    with open(filename, 'wb') as f:
        pickle.dump(params, f)

def load_model(filename):
    """Load model parameters from a file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Save the model after training
print("\nSaving model weights...")
save_model(params, 'model_weights.pkl')

def generate(params, prompt_tokens, max_new_tokens, args, temperature=0.7):
    """Improved generation function"""
    key = random.PRNGKey(int(time.time() * 1000))
    x = jnp.array(prompt_tokens)[None, :]  # [1, seq_len]
    
    for _ in range(max_new_tokens):
        # Get logits and ignore cache
        logits, _ = model_forward(params, x, args)
        
        # Get logits of last token and apply temperature
        logits = logits[0, -1, :] / temperature
        
        # Sample from the distribution
        key, subkey = random.split(key)
        next_token = random.categorical(subkey, logits[None, :])[0]
        
        # Append to the sequence
        x = jnp.concatenate([x, next_token[None, None]], axis=1)
    
    return x[0]

# Test the model with more interesting prompts
print("\nGenerating sample texts...")
prompts = [
    "O, fair maiden, thy beauty doth outshine the stars. Speak, and let my heart be thine.",
    "To love or not to love, that is the question. What is the nature of the heart's desire?",
    "Alas, my fate is sealed, and the shadows of despair loom over me. What hope remains in this dark hour?"  
]

for prompt in prompts:
    print("\nPrompt:", prompt)
    prompt_tokens = enc.encode(prompt)
    generated_tokens = generate(params, prompt_tokens, 100, args, temperature=0.8)
    generated_text = enc.decode(generated_tokens.tolist())
    print("Generated:", generated_text)
