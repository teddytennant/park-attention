"""Numerical checks for PAMA, QRKC, top-p retrieval, and sparse attention."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from park_attention import (
    attention_recall,
    centroid_average_importance,
    dense_attention,
    disagreement_case,
    exact_block_importance,
    pama,
    qrkc,
    sparse_attention,
    token_mask,
    top_p_block_mask,
)


def _assert_close(actual, expected, rtol=1e-6, atol=1e-6, message=""):
    actual = jnp.asarray(actual)
    expected = jnp.asarray(expected)
    if bool(jnp.allclose(actual, expected, rtol=rtol, atol=atol)):
        return
    diff = float(jnp.max(jnp.abs(actual - expected)))
    raise AssertionError(f"{message} max abs diff {diff}")


def test_pama_rows_sum_to_one():
    key = jax.random.PRNGKey(0)
    q_key, c_key = jax.random.split(key)
    queries = jax.random.normal(q_key, (7, 5), dtype=jnp.float64)
    centroids = jax.random.normal(c_key, (4, 5), dtype=jnp.float64)
    sizes = jnp.array([1.0, 2.0, 5.0, 3.0], dtype=jnp.float64)
    query_block_ids = jnp.array([0, 1, 1, 0, 2, 2, 0], dtype=jnp.int32)
    importance = pama(queries, centroids, sizes, query_block_ids)
    _assert_close(jnp.sum(importance, axis=-1), jnp.ones(3, dtype=jnp.float64), atol=1e-8)


def test_pama_matches_exact_when_keys_inside_a_block_are_identical():
    key = jax.random.PRNGKey(1)
    c_key, q_key, v_key = jax.random.split(key, 3)
    dim = 6
    centroids = jax.random.normal(c_key, (3, dim), dtype=jnp.float64)
    sizes = (2, 5, 1)
    keys = jnp.concatenate(
        [jnp.repeat(centroids[v][None, :], sizes[v], axis=0) for v in range(3)],
        axis=0,
    )
    key_block_ids = jnp.array([0, 0, 1, 1, 1, 1, 1, 2], dtype=jnp.int32)
    queries = jax.random.normal(q_key, (6, dim), dtype=jnp.float64)
    query_block_ids = jnp.array([1, 0, 1, 0, 1, 0], dtype=jnp.int32)
    values = jax.random.normal(v_key, (keys.shape[0], dim), dtype=jnp.float64)
    _, attention = dense_attention(queries, keys, values)
    # Centroids passed to PAMA are the keys themselves, recomputed as means.
    means = jnp.stack([jnp.mean(keys[key_block_ids == v], axis=0) for v in range(3)])
    got = pama(queries, means, jnp.array(sizes, dtype=jnp.float64), query_block_ids)
    expected = exact_block_importance(attention, query_block_ids, key_block_ids)
    _assert_close(got, expected, atol=1e-8, rtol=1e-8)
    _assert_close(jnp.sum(got, axis=-1), jnp.ones(2), atol=1e-8)
    _assert_close(jnp.sum(expected, axis=-1), jnp.ones(2), atol=1e-8)


def test_qrkc_transform_matches_cholesky_and_metric_norm():
    key = jax.random.PRNGKey(2)
    q_key, k_key, v_key = jax.random.split(key, 3)
    length, dim = 8, 4
    queries = jax.random.normal(q_key, (length, dim), dtype=jnp.float64)
    keys = jax.random.normal(k_key, (length, dim), dtype=jnp.float64)
    values = jax.random.normal(v_key, (length, dim), dtype=jnp.float64)
    alpha = 0.05
    eps = 1e-6
    result = qrkc(queries, keys, values, n_clusters=3, n_iters=6, alpha=alpha, eps=eps)

    gram = (queries.T @ queries) / length
    gram = (gram + gram.T) / 2
    scale = jnp.maximum(jnp.trace(gram) / dim, 1e-12)
    eye = jnp.eye(dim, dtype=jnp.float64)
    shrunk = (1.0 - alpha) * gram + alpha * scale * eye + eps * scale * eye

    _assert_close(result.cholesky_factor @ result.cholesky_factor.T, shrunk, atol=1e-10)
    _assert_close(result.metric, shrunk, atol=1e-12)
    _assert_close(result.transformed_keys, keys @ result.cholesky_factor, atol=1e-12)
    assert bool(jnp.all(jnp.isfinite(result.centroids)))
    assert bool(jnp.all(result.sizes > 0))
    assert bool(jnp.array_equal(result.permutation, jnp.argsort(result.labels)))
    assert bool(jnp.array_equal(result.keys, keys[result.permutation]))
    assert bool(jnp.array_equal(result.values, values[result.permutation]))

    transformed = result.transformed_keys
    sse = 0.0
    metric_norm = 0.0
    raw_norm = 0.0
    for cluster in range(3):
        mask = result.labels == cluster
        transformed_mean = jnp.mean(transformed[mask], axis=0)
        original_mean = jnp.mean(keys[mask], axis=0)
        _assert_close(result.centroids[cluster], original_mean, atol=1e-10)
        residual = transformed[mask] - transformed_mean
        sse = sse + jnp.sum(residual * residual)
        delta = keys[mask] - original_mean
        metric_norm = metric_norm + jnp.sum(delta * (delta @ shrunk))
        raw_norm = raw_norm + jnp.sum(delta * (delta @ gram))
    _assert_close(sse, metric_norm, atol=1e-8, rtol=1e-8, message="shrunk G_Q norm")
    # Shrinkage is not a no-op on this draw, so the unshrunk norm must differ.
    assert float(jnp.abs(sse - raw_norm)) > float(jnp.abs(sse - metric_norm)) + 1e-6


def test_cholesky_identity_on_rank_deficient_queries():
    key = jax.random.PRNGKey(3)
    queries = jax.random.normal(key, (3, 5), dtype=jnp.float64)
    keys = jax.random.normal(jax.random.fold_in(key, 1), (3, 5), dtype=jnp.float64)
    values = jnp.ones_like(keys)
    result = qrkc(queries, keys, values, n_clusters=2, n_iters=2)
    rebuilt = result.cholesky_factor @ result.cholesky_factor.T
    _assert_close(rebuilt, result.metric, atol=1e-10)
    assert bool(jnp.all(jnp.isfinite(result.centroids)))
    assert bool(jnp.all(jnp.isfinite(result.transformed_keys)))


def test_qrkc_reseed_leaves_finite_centroids_when_keys_are_identical():
    queries = jnp.ones((6, 3), dtype=jnp.float64)
    keys = jnp.ones((6, 3), dtype=jnp.float64)
    values = jnp.ones((6, 3), dtype=jnp.float64)
    result = qrkc(queries, keys, values, n_clusters=4, n_iters=3)
    assert bool(jnp.all(jnp.isfinite(result.centroids)))
    assert bool(jnp.all(jnp.isfinite(result.metric)))
    assert bool(jnp.all(jnp.isfinite(result.cholesky_factor)))
    assert bool(jnp.all(jnp.isfinite(result.keys)))


def test_full_mask_sparse_attention_matches_dense():
    key = jax.random.PRNGKey(4)
    q_key, k_key, v_key = jax.random.split(key, 3)
    queries = jax.random.normal(q_key, (5, 7), dtype=jnp.float64)
    keys = jax.random.normal(k_key, (6, 7), dtype=jnp.float64)
    values = jax.random.normal(v_key, (6, 3), dtype=jnp.float64)
    dense_out, dense_weights = dense_attention(queries, keys, values)
    mask = jnp.ones((5, 6), dtype=bool)
    sparse_out, sparse_weights = sparse_attention(queries, keys, values, mask)
    _assert_close(sparse_weights, dense_weights, atol=1e-12)
    _assert_close(sparse_out, dense_out, atol=1e-12)


def test_partial_mask_renormalizes_over_kept_keys():
    queries = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    keys = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float64)
    values = jnp.array([[1.0], [2.0], [3.0]], dtype=jnp.float64)
    mask = jnp.array([[True, False, True], [False, True, False]])
    _, weights = sparse_attention(queries, keys, values, mask)
    assert bool(weights[0, 1] == 0.0)
    assert bool(weights[1, 0] == 0.0)
    assert bool(weights[1, 2] == 0.0)
    _assert_close(jnp.sum(weights, axis=-1), jnp.ones(2), atol=1e-12)


def test_disagreement_pama_beats_centroid_average():
    case = disagreement_case(copies=4)
    pama_importance = pama(
        case.queries, case.key_centroids, case.key_sizes, case.query_block_ids
    )
    centroid_importance = centroid_average_importance(
        case.queries, case.key_centroids, case.key_sizes, case.query_block_ids
    )
    _, attention = dense_attention(case.queries, case.keys, case.values)
    exact = exact_block_importance(attention, case.query_block_ids, case.key_block_ids)
    # Keys are identical inside each block, so PAMA is the exact block importance.
    _assert_close(pama_importance, exact, atol=1e-6, rtol=1e-6)
    assert float(jnp.max(jnp.abs(centroid_importance - exact))) > 0.2

    threshold = 0.9
    pama_blocks = top_p_block_mask(pama_importance, threshold)
    centroid_blocks = top_p_block_mask(centroid_importance, threshold)
    assert bool(pama_blocks[0, 0]) and bool(pama_blocks[0, 1])
    assert not bool(pama_blocks[0, 2])
    assert bool(centroid_blocks[0, 2])
    assert not bool(centroid_blocks[0, 0])
    assert not bool(centroid_blocks[0, 1])

    pama_mask = token_mask(pama_blocks, case.query_block_ids, case.key_block_ids)
    centroid_mask = token_mask(centroid_blocks, case.query_block_ids, case.key_block_ids)
    pama_recall = attention_recall(attention, pama_mask)
    centroid_recall = attention_recall(attention, centroid_mask)
    assert float(pama_recall) > float(centroid_recall)
    assert float(pama_recall) > 0.95
    assert float(centroid_recall) < 0.2


def test_centroid_baseline_matches_pama_when_queries_agree():
    dim = 4
    queries = jnp.array([[0.2, -0.4, 0.1, 0.3]], dtype=jnp.float64)
    queries = jnp.repeat(queries, 3, axis=0)
    centroids = jnp.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5]],
        dtype=jnp.float64,
    )
    sizes = jnp.array([2.0, 1.0, 4.0], dtype=jnp.float64)
    ids = jnp.zeros((3,), dtype=jnp.int32)
    _assert_close(
        pama(queries, centroids, sizes, ids),
        centroid_average_importance(queries, centroids, sizes, ids),
        atol=1e-12,
    )


def test_top_p_one_keeps_every_block():
    importance = jnp.array(
        [[0.7, 0.3, 0.0], [0.2, 0.2, 0.6], [1.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    kept = top_p_block_mask(importance, p=1.0)
    assert bool(jnp.all(kept))
    query_ids = jnp.array([0, 1, 2, 0], dtype=jnp.int32)
    key_ids = jnp.array([2, 0, 1], dtype=jnp.int32)
    mask = token_mask(kept, query_ids, key_ids)
    assert bool(jnp.all(mask))


def test_top_p_stops_when_cumulative_mass_reaches_threshold():
    importance = jnp.array([[0.2, 0.5, 0.3]], dtype=jnp.float64)
    kept = top_p_block_mask(importance, p=0.7)
    # Descending order is block 1 (0.5), block 2 (0.3), block 0 (0.2).
    # 0.5 is not enough. 0.5 + 0.3 reaches 0.8, so blocks 1 and 2 are kept.
    assert bool(kept[0, 1]) and bool(kept[0, 2])
    assert not bool(kept[0, 0])
