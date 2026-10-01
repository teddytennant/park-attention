"""Per-query Attention-Mass Aggregation, paper equations (8) and (9)."""

from jax import Array
import jax
import jax.numpy as jnp


def _as_2d(name: str, value) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (tokens, dim), got {array.shape}")
    return array


def _query_block_count(query_block_ids: Array, n_query_blocks: int | None) -> int:
    if n_query_blocks is None:
        return int(jnp.max(query_block_ids)) + 1
    return int(n_query_blocks)


def per_query_block_weights(
    queries,
    key_centroids,
    key_sizes,
) -> Array:
    """Softmax over key blocks for every original query.

    S_iv = q_i · c_v / sqrt(d) + log(n_v), then P_iv = softmax_v(S_iv).
    The log(n_v) term is the block-size correction in equation (8). It is
    equivalent to weighting exp(q_i · c_v / sqrt(d)) by n_v and renormalizing.
    """
    queries = _as_2d("queries", queries)
    key_centroids = _as_2d("key_centroids", key_centroids)
    if queries.shape[-1] != key_centroids.shape[-1]:
        raise ValueError(
            "queries and key_centroids must share the head dimension, "
            f"got {queries.shape[-1]} and {key_centroids.shape[-1]}"
        )
    dtype = queries.dtype
    sizes = jnp.asarray(key_sizes, dtype=dtype)
    if sizes.shape != (key_centroids.shape[0],):
        raise ValueError(
            "key_sizes must have one entry per key block, "
            f"got {sizes.shape} for {key_centroids.shape[0]} centroids"
        )
    scale = jnp.sqrt(jnp.array(queries.shape[-1], dtype=dtype))
    logits = (queries @ key_centroids.T) / scale + jnp.log(sizes)[None, :]
    return jax.nn.softmax(logits, axis=-1)


def _average_within_query_blocks(
    token_weights: Array,
    query_block_ids: Array,
    n_query_blocks: int,
) -> Array:
    """Average per-query distributions inside each query block. Rows sum to 1."""
    groups = jax.nn.one_hot(query_block_ids, n_query_blocks, dtype=token_weights.dtype)
    counts = jnp.sum(groups, axis=0)
    if bool(jnp.any(counts <= 0)):
        raise ValueError("every query block id from 0 to n_query_blocks-1 must occur")
    summed = groups.T @ token_weights
    return summed / counts[:, None]


def pama(queries, key_centroids, key_sizes, query_block_ids, n_query_blocks=None) -> Array:
    """Block importance W from equations (8) and (9).

    Every original query is kept. Softmax runs over key blocks, then those
    distributions are averaged inside the query block. This is not "average
    the queries, then softmax".

    Returns W with shape (n_query_blocks, n_key_blocks). Each row sums to 1.
    """
    weights = per_query_block_weights(queries, key_centroids, key_sizes)
    ids = jnp.asarray(query_block_ids, dtype=jnp.int32)
    if ids.shape != (weights.shape[0],):
        raise ValueError(
            "query_block_ids must have one id per query, "
            f"got {ids.shape} for {weights.shape[0]} queries"
        )
    n_blocks = _query_block_count(ids, n_query_blocks)
    return _average_within_query_blocks(weights, ids, n_blocks)


def centroid_average_importance(
    queries,
    key_centroids,
    key_sizes,
    query_block_ids,
    n_query_blocks=None,
) -> Array:
    """Centroid baseline: average queries first, then softmax against key centroids.

    This is the query-side aggregation mismatch PARK avoids. The key-centroid
    score still uses equation (8), including log(n_v), so the only change is
    the order of averaging and softmax. With equal key-block sizes the log
    term is a constant shift and does not change the softmax.
    """
    queries = _as_2d("queries", queries)
    key_centroids = _as_2d("key_centroids", key_centroids)
    ids = jnp.asarray(query_block_ids, dtype=jnp.int32)
    if ids.shape != (queries.shape[0],):
        raise ValueError(
            "query_block_ids must have one id per query, "
            f"got {ids.shape} for {queries.shape[0]} queries"
        )
    n_blocks = _query_block_count(ids, n_query_blocks)
    groups = jax.nn.one_hot(ids, n_blocks, dtype=queries.dtype)
    counts = jnp.sum(groups, axis=0)
    if bool(jnp.any(counts <= 0)):
        raise ValueError("every query block id from 0 to n_query_blocks-1 must occur")
    query_means = (groups.T @ queries) / counts[:, None]
    return per_query_block_weights(query_means, key_centroids, key_sizes)
