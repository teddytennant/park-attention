"""Synthetic query-block disagreement case.

Two query groups share one block and attend to different key blocks. Their
vector average aligns with a third, distractor key block. Averaging queries
before softmax therefore retrieves the wrong block. PAMA keeps each query's
distribution and retrieves both true blocks.
"""

from typing import NamedTuple

from jax import Array
import jax.numpy as jnp


class DisagreementCase(NamedTuple):
    queries: Array
    keys: Array
    values: Array
    query_block_ids: Array
    key_block_ids: Array
    key_centroids: Array
    key_sizes: Array
    true_key_blocks: Array
    distractor_block: int


def disagreement_case(copies: int = 4) -> DisagreementCase:
    """One query block, three equal key blocks, keys identical inside each block.

    Key block 0 matches the first query group, block 1 matches the second, and
    block 2 matches the mean query. copies is the number of identical keys in
    each block and the number of queries in each group.
    """
    if copies < 1:
        raise ValueError(f"copies must be at least 1, got {copies}")
    # Large enough that the mean query's softmax mass on the distractor exceeds
    # the paper's top-p threshold of 0.9, while each original query stays peaked
    # on its own key block.
    signal = jnp.array(16.0)
    half = jnp.array(0.5)
    zero = jnp.array(0.0)
    query_a = jnp.stack([half, signal])
    query_b = jnp.stack([half, -signal])
    centroid_a = query_a
    centroid_b = query_b
    centroid_distractor = jnp.stack([signal, zero])

    queries = jnp.concatenate(
        [jnp.repeat(query_a[None, :], copies, axis=0), jnp.repeat(query_b[None, :], copies, axis=0)],
        axis=0,
    )
    keys = jnp.concatenate(
        [
            jnp.repeat(centroid_a[None, :], copies, axis=0),
            jnp.repeat(centroid_b[None, :], copies, axis=0),
            jnp.repeat(centroid_distractor[None, :], copies, axis=0),
        ],
        axis=0,
    )
    values = jnp.arange(keys.shape[0] * keys.shape[1], dtype=queries.dtype).reshape(keys.shape)
    query_block_ids = jnp.zeros((queries.shape[0],), dtype=jnp.int32)
    key_block_ids = jnp.concatenate(
        [
            jnp.full((copies,), 0, dtype=jnp.int32),
            jnp.full((copies,), 1, dtype=jnp.int32),
            jnp.full((copies,), 2, dtype=jnp.int32),
        ]
    )
    key_centroids = jnp.stack([centroid_a, centroid_b, centroid_distractor])
    key_sizes = jnp.full((3,), copies, dtype=queries.dtype)
    return DisagreementCase(
        queries=queries,
        keys=keys,
        values=values,
        query_block_ids=query_block_ids,
        key_block_ids=key_block_ids,
        key_centroids=key_centroids,
        key_sizes=key_sizes,
        true_key_blocks=jnp.array([0, 1], dtype=jnp.int32),
        distractor_block=2,
    )
