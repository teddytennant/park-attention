"""Dense attention (equation 1) and mask-sparse attention."""

from jax import Array
import jax
import jax.numpy as jnp


def _as_2d(name: str, value) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (tokens, dim), got {array.shape}")
    return array


def attention_logits(queries, keys) -> Array:
    """Q K^T / sqrt(d), equation (1)."""
    queries = _as_2d("queries", queries)
    keys = _as_2d("keys", keys)
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError(
            "queries and keys must share the head dimension, "
            f"got {queries.shape[-1]} and {keys.shape[-1]}"
        )
    scale = jnp.sqrt(jnp.array(queries.shape[-1], dtype=queries.dtype))
    return (queries @ keys.T) / scale


def dense_attention(queries, keys, values):
    """Softmax(Q K^T / sqrt(d)) V. Returns output and the attention matrix."""
    values = _as_2d("values", values)
    logits = attention_logits(queries, keys)
    if values.shape[0] != logits.shape[1]:
        raise ValueError(
            "values must have one row per key, "
            f"got {values.shape[0]} values and {logits.shape[1]} keys"
        )
    attention = jax.nn.softmax(logits, axis=-1)
    return attention @ values, attention


def sparse_attention(queries, keys, values, mask):
    """Softmax only over keys where mask is true, then multiply by V.

    Masked keys are set to -inf before softmax, so each query renormalizes over
    the keys it keeps. A mask of all ones matches dense attention.
    """
    values = _as_2d("values", values)
    logits = attention_logits(queries, keys)
    keep = jnp.asarray(mask).astype(bool)
    if keep.shape != logits.shape:
        raise ValueError(
            f"mask must have shape {logits.shape}, got {keep.shape}"
        )
    masked_logits = jnp.where(keep, logits, jnp.array(-jnp.inf, dtype=logits.dtype))
    attention = jax.nn.softmax(masked_logits, axis=-1)
    attention = jnp.where(keep, attention, jnp.array(0.0, dtype=attention.dtype))
    return attention @ values, attention


def attention_recall(attention, mask) -> Array:
    """Equation (4): sum(A * M) / sum(A), with A the dense post-softmax weights."""
    weights = jnp.asarray(attention)
    covered = jnp.asarray(mask, dtype=weights.dtype)
    if covered.shape != weights.shape:
        raise ValueError(
            f"mask must have shape {weights.shape}, got {covered.shape}"
        )
    total = jnp.sum(weights)
    return jnp.sum(weights * covered) / total
