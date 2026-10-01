"""Exact block importance (equation 3) and top-p block retrieval."""

from jax import Array
import jax
import jax.numpy as jnp


def exact_block_importance(attention, query_block_ids, key_block_ids) -> Array:
    """Equation (3): average, over queries in the block, of attention mass on a key block.

    W_uv = (1 / n_u) * sum_{i in Q_u} sum_{j in K_v} A_ij.
    Rows sum to 1 when A is a row-stochastic attention matrix and the key
    blocks partition the keys.
    """
    weights = jnp.asarray(attention)
    if weights.ndim != 2:
        raise ValueError(f"attention must have shape (queries, keys), got {weights.shape}")
    query_ids = jnp.asarray(query_block_ids, dtype=jnp.int32)
    key_ids = jnp.asarray(key_block_ids, dtype=jnp.int32)
    if query_ids.shape != (weights.shape[0],):
        raise ValueError("query_block_ids must have one id per query")
    if key_ids.shape != (weights.shape[1],):
        raise ValueError("key_block_ids must have one id per key")
    n_query_blocks = int(jnp.max(query_ids)) + 1
    n_key_blocks = int(jnp.max(key_ids)) + 1
    query_groups = jax.nn.one_hot(query_ids, n_query_blocks, dtype=weights.dtype)
    key_groups = jax.nn.one_hot(key_ids, n_key_blocks, dtype=weights.dtype)
    query_counts = jnp.sum(query_groups, axis=0)
    if bool(jnp.any(query_counts <= 0)):
        raise ValueError("every query block id from 0 to n_query_blocks-1 must occur")
    block_sum = query_groups.T @ weights @ key_groups
    return block_sum / query_counts[:, None]


def top_p_block_mask(importance, p: float) -> Array:
    """Keep key blocks in descending importance until the cumulative mass reaches p.

    importance has shape (n_query_blocks, n_key_blocks) and is read row by row.
    p >= 1 keeps every block, including blocks whose estimated importance is 0.
    The block that crosses the threshold is kept.
    """
    weights = jnp.asarray(importance)
    if weights.ndim != 2:
        raise ValueError(
            f"importance must have shape (query blocks, key blocks), got {weights.shape}"
        )
    n_query, n_key = weights.shape
    order = jnp.argsort(-weights, axis=-1)
    sorted_weights = jnp.take_along_axis(weights, order, axis=-1)
    cumulative = jnp.cumsum(sorted_weights, axis=-1)
    before = jnp.concatenate(
        [jnp.zeros((n_query, 1), dtype=weights.dtype), cumulative[:, :-1]],
        axis=-1,
    )
    keep_sorted = before < jnp.asarray(p, dtype=weights.dtype)
    keep_sorted = jnp.where(jnp.asarray(p) >= 1, True, keep_sorted)
    keep = jnp.zeros((n_query, n_key), dtype=bool)
    rows = jnp.arange(n_query)[:, None]
    return keep.at[rows, order].set(keep_sorted)


def token_mask(block_keep, query_block_ids, key_block_ids) -> Array:
    """Token-level mask from kept query-block / key-block pairs."""
    keep = jnp.asarray(block_keep).astype(bool)
    query_ids = jnp.asarray(query_block_ids, dtype=jnp.int32)
    key_ids = jnp.asarray(key_block_ids, dtype=jnp.int32)
    return keep[query_ids[:, None], key_ids[None, :]]
