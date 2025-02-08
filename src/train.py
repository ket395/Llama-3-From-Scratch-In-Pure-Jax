import jax
import jax.numpy as jnp
from jax import random
import tiktoken
from functools import partial
from model import init_model_params, model_forward
import os
import pickle

os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
print("JAX devices:", jax.devices())

# Initialize tokenizer and load data
enc = tiktoken.get_encoding("gpt2")
with open('shakespeare.txt', 'r') as f:
    text = f.read()
tokens = enc.encode(text)
data = jnp.array(tokens)

# Model configuration
class ModelConfig:
    vocab_size = enc.n_vocab  
    dim = 256  
    n_layers = 6  
    n_heads = 8 
    n_kv_heads = 4  
    max_seq_len = 512
    batch_size = 32  
    learning_rate = 3e-4
    dropout_rate = 0.0

config = ModelConfig()

# Initialize model
key = random.PRNGKey(0)
params = init_model_params(
    key=key,
    vocab_size=config.vocab_size,
    dim=config.dim,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    n_kv_heads=config.n_kv_heads
)

def save_params(params, filepath):
    numpy_params = jax.tree_map(lambda x: x.copy(), params)
    with open(filepath, 'wb') as f:
        pickle.dump(numpy_params, f)

def load_params(filepath):
    with open(filepath, 'rb') as f:
        numpy_params = pickle.load(f)
    # convert back to JAX arrays
    params = jax.tree_map(lambda x: jnp.array(x), numpy_params)
    return params

def get_batch(key, data, batch_size, seq_len):
    ix = random.randint(key, (batch_size,), 0, len(data) - seq_len)
    x = jnp.stack([data[i:i+seq_len] for i in ix])
    y = jnp.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

def generate(params, prompt_tokens, max_new_tokens, config):
    x = jnp.array(prompt_tokens)
    for _ in range(max_new_tokens):
        x_crop = x[-config.max_seq_len:]
        logits, _ = model_forward(params, x_crop[None, :], config)
        logits = logits[0, -1, :]  # take the last logit
        next_token = random.categorical(random.PRNGKey(0), logits, shape=(1,))[0]
        x = jnp.concatenate([x, jnp.array([next_token])])
    return x.tolist()

def compute_loss(params, batch):
    inputs, targets = batch
    logits, _ = model_forward(params, inputs, config)
    logits = logits.reshape(-1, config.vocab_size)
    targets = targets.reshape(-1)
    loss = -jnp.mean(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits),
            targets[:, None],
            axis=1
        )
    )
    return loss

@jax.jit
def update_step(params, batch):
    loss, grads = jax.value_and_grad(compute_loss)(params, batch)
    params = jax.tree.map(
        lambda p, g: p - config.learning_rate * g,
        params,
        grads
    )
    return params, loss

def train(num_epochs=30, steps_per_epoch=1000):
    key = random.PRNGKey(0)
    params_state = params  # copying
    
    epoch_losses = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            
            key, batch_key = random.split(key)
            
            # Get batch
            batch = get_batch(batch_key, data, config.batch_size, config.max_seq_len)
            
            # Update model
            params_state, loss = update_step(params_state, batch)
            epoch_loss += loss
            
            
            if step % 100 == 0:
                print(f"epoch {epoch + 1}, step {step}/{steps_per_epoch}: loss = {loss:.4f}")
        
       
        avg_epoch_loss = epoch_loss / steps_per_epoch
        epoch_losses.append(avg_epoch_loss)
        
        print(f"\nepoch {epoch + 1} | average loss: {avg_epoch_loss:.4f}")
        
        
        if (epoch + 1) % 5 == 0:
            save_params(params_state, f'model_checkpoint_epoch_{epoch+1}.pkl')
    
   
    print("Loss by epoch:")
    for epoch, loss in enumerate(epoch_losses, 1):
        print(f"Epoch {epoch}: {loss:.4f}")
    
    # Save final model
    save_params(params_state, 'model_final.pkl')
    return params_state

# Train the model
trained_params = train()
