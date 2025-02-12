import jax
import jax.numpy as jnp
from jax import random
import math

def rms_norm(x, weight, eps=1e-5):
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * weight * jnp.reciprocal(jnp.sqrt(variance + eps))

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim // 2, dtype=jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    return jnp.complex64(jnp.exp(1j * freqs))

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_r, xk_r = jnp.reshape(xq, (*xq.shape[:-1], -1, 2)), jnp.reshape(xk, (*xk.shape[:-1], -1, 2))
    xq_complex = jnp.complex64(xq_r[..., 0] + 1j * xq_r[..., 1])
    xk_complex = jnp.complex64(xk_r[..., 0] + 1j * xk_r[..., 1])
    freqs_cis = jnp.reshape(freqs_cis, (1, freqs_cis.shape[0], 1, freqs_cis.shape[1]))
    xq_out = xq_complex * freqs_cis
    xk_out = xk_complex * freqs_cis
    xq = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1).reshape(xq.shape)
    xk = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1).reshape(xk.shape)
    return xq, xk

def repeat_kv(x, n_rep):
    return x if n_rep == 1 else jnp.repeat(x, n_rep, axis=2)

def init_weight(key, shape, scale=None):
    scale = 1.0 / math.sqrt(shape[0]) if scale is None else scale
    return jax.random.normal(key, shape) * scale

def init_attention_weights(key, dim, n_heads, n_kv_heads):
    keys = jax.random.split(key, 4)
    head_dim = dim // n_heads
    return {
        'wq': init_weight(keys[0], (dim, n_heads * head_dim)),
        'wk': init_weight(keys[1], (dim, n_kv_heads * head_dim)),
        'wv': init_weight(keys[2], (dim, n_kv_heads * head_dim)),
        'wo': init_weight(keys[3], (n_heads * head_dim, dim))
    }
def init_ffn_weights(key, dim):
    keys = jax.random.split(key, 3)
    return {
        'w1': init_weight(keys[0], (dim, 4 * dim)),
        'w2': init_weight(keys[1], (4 * dim, dim)),
        'w3': init_weight(keys[2], (dim, 4 * dim))}
    
def init_transformer_block(key, dim, n_heads, n_kv_heads):
    keys = jax.random.split(key, 4)
    return {
        'attention': init_attention_weights(keys[0], dim, n_heads, n_kv_heads),
        'ffn': init_ffn_weights(keys[1], dim),
        'attention_norm': init_weight(keys[2], (dim,), scale=1.0),
        'ffn_norm': init_weight(keys[3], (dim,), scale=1.0)}
    
def init_model_params(key, vocab_size, dim, n_layers, n_heads, n_kv_heads):
    keys = jax.random.split(key, 4)
    params = {
        'token_embedding': init_weight(keys[0], (vocab_size, dim)),
        'norm_f': init_weight(keys[1], (dim,), scale=1.0),
        'output': init_weight(keys[2], (dim, vocab_size))
    }
    block_keys = jax.random.split(keys[3], n_layers)
    params['blocks'] = [init_transformer_block(k, dim, n_heads, n_kv_heads) for k in block_keys]
    return params

def attention(params, x, mask, freqs_cis, n_heads, n_kv_heads, cache=None, position=0):
    B, T, C = x.shape
    head_dim = C // n_heads
    q = jnp.dot(x, params['wq']).reshape(B, T, n_heads, head_dim)
    k = jnp.dot(x, params['wk']).reshape(B, T, n_kv_heads, head_dim)
    v = jnp.dot(x, params['wv']).reshape(B, T, n_kv_heads, head_dim)
    q, k = apply_rotary_emb(q, k, freqs_cis[position:position + T])
    if cache is not None:
        k = jnp.concatenate([cache[0], k], axis=1)
        v = jnp.concatenate([cache[1], v], axis=1)
    new_cache = (k, v)
    k = repeat_kv(k, n_heads // n_kv_heads)
    v = repeat_kv(v, n_heads // n_kv_heads)
    q, k, v = map(lambda x: x.transpose(0, 2, 1, 3), (q, k, v))
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores + mask[:, :, :T, :T]
    scores = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(scores, v)
    output = output.transpose(0, 2, 1, 3).reshape(B, T, -1)
    return jnp.dot(output, params['wo']), new_cache

def feed_forward(params, x):
    return jnp.dot(jax.nn.silu(jnp.dot(x, params['w3'])) * jnp.dot(x, params['w1']), params['w2'])

def transformer_block(params, x, mask, freqs_cis, n_heads, n_kv_heads, cache=None, position=0, training=False, dropout_rate=0.0, key=None):
    attn_output, new_cache = attention(params['attention'], rms_norm(x, params['attention_norm']), mask, freqs_cis, n_heads, n_kv_heads, cache, position)
    if training:
        dropout_key, key = jax.random.split(key)
        attn_output = jax.random.bernoulli(dropout_key, 1-dropout_rate, shape=attn_output.shape) * attn_output / (1-dropout_rate)
    h = x + attn_output
    ffn_output = feed_forward(params['ffn'], rms_norm(h, params['ffn_norm']))
    if training:
        dropout_key, key = jax.random.split(key)
        ffn_output = jax.random.bernoulli(dropout_key, 1-dropout_rate, shape=ffn_output.shape) * ffn_output / (1-dropout_rate)
    out = h + ffn_output
    return out, new_cache

def model_forward(params, inputs, config, cache=None, position=0):
    B, T = inputs.shape
    h = params['token_embedding'][inputs]
    freqs_cis = precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len)
    mask = jnp.tril(jnp.ones((config.max_seq_len, config.max_seq_len)))
    mask = jnp.where(mask == 0, -1e9, 0.0)
    mask = mask.astype(h.dtype)
    mask = mask[None, None, :, :]
    new_caches = []
    for i, block in enumerate(params['blocks']):
        layer_cache = cache[i] if cache is not None else None
        h, layer_cache = transformer_block(block, h, mask, freqs_cis, config.n_heads, config.n_kv_heads, layer_cache, position, training=False, dropout_rate=config.dropout_rate)
        new_caches.append(layer_cache)
    h = rms_norm(h, params['norm_f'])
    logits = jnp.dot(h, params['output'])
    return logits, new_caches