"""Query-Response Key Clustering, Appendix F Algorithm 3."""

from typing import NamedTuple

from jax import Array
import jax
import jax.numpy as jnp

# Paper section on numerically stable Cholesky: alpha = 0.05, s_min = 1e-12.
# The stability constant eps is described only as a small positive number.
DEFAULT_ALPHA = 0.05
DEFAULT_EPS = 1e-6
S_MIN = 1e-12


class QRKCResult(NamedTuple):
    """Labels are in the original token order. Keys and values are permuted."""

    labels: Array
    permutation: Array
    keys: Array
    values: Array
    centroids: Array
    sizes: Array
    cholesky_factor: Array
    metric: Array
    transformed_keys: Array


def shrunk_query_metric(queries, alpha: float = DEFAULT_ALPHA, eps: float = DEFAULT_EPS) -> Array:
    """Stabilized query metric from Algorithm 3 steps 1-4.

    G = Q^T Q / L, symmetrized, then
    G <- (1 - alpha) G + alpha * s_Q * I + eps * s_Q * I,
    with s_Q = max(trace(G) / d, 1e-12).
    """
    queries = jnp.asarray(queries)
    if queries.ndim != 2:
        raise ValueError(f"queries must have shape (L, d), got {queries.shape}")
    length, dim = queries.shape
    if length < 1 or dim < 1:
        raise ValueError(f"queries must be non-empty, got {queries.shape}")
    gram = (queries.T @ queries) / jnp.array(length, dtype=queries.dtype)
    gram = (gram + gram.T) / 2
    scale = jnp.maximum(
        jnp.trace(gram) / jnp.array(dim, dtype=queries.dtype),
        jnp.array(S_MIN, dtype=queries.dtype),
    )
    eye = jnp.eye(dim, dtype=queries.dtype)
    alpha_f = jnp.array(alpha, dtype=queries.dtype)
    eps_f = jnp.array(eps, dtype=queries.dtype)
    return (1.0 - alpha_f) * gram + alpha_f * scale * eye + eps_f * scale * eye


def _squared_distances(points: Array, centroids: Array) -> Array:
    diff = points[:, None, :] - centroids[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def _init_farthest(points: Array, n_clusters: int) -> Array:
    """Deterministic farthest-point init drawn from the transformed keys."""
    scores = jnp.sum(points * points, axis=-1)
    chosen = []
    for _ in range(n_clusters):
        index = jnp.argmax(scores)
        chosen.append(points[index])
        distance = jnp.sum((points - points[index]) ** 2, axis=-1)
        scores = jnp.minimum(scores, distance)
        scores = scores.at[index].set(-jnp.inf)
    return jnp.stack(chosen, axis=0)


def _cluster_sums(points: Array, labels: Array, n_clusters: int):
    groups = jax.nn.one_hot(labels, n_clusters, dtype=points.dtype)
    counts = jnp.sum(groups, axis=0)
    sums = groups.T @ points
    return sums, counts


def _means_from_sums(sums: Array, counts: Array) -> Array:
    safe = jnp.maximum(counts[:, None], jnp.array(1.0, dtype=sums.dtype))
    return sums / safe


def _farthest_order(points: Array, labels: Array, centroids: Array) -> Array:
    """Indices of transformed keys, farthest from the assigned centroid first."""
    distances = _squared_distances(points, centroids)
    assigned = distances[jnp.arange(points.shape[0]), labels]
    return jnp.argsort(-assigned)


def _fill_empty(points: Array, means: Array, counts: Array, farthest_order: Array) -> Array:
    """Write farthest keys into empty centroids. Non-empty means are kept."""
    empty = counts == 0
    empty_rank = jnp.cumsum(empty.astype(jnp.int32)) - 1
    take = jnp.clip(empty_rank, 0, points.shape[0] - 1)
    reseed = points[farthest_order[take]]
    return jnp.where(empty[:, None], reseed, means)


def _lloyd(transformed: Array, n_clusters: int, n_iters: int) -> Array:
    """Lloyd iterations. Labels come from the last assignment, as in Algorithm 3."""
    centroids = _init_farthest(transformed, n_clusters)
    labels = jnp.zeros((transformed.shape[0],), dtype=jnp.int32)
    for _ in range(n_iters):
        distances = _squared_distances(transformed, centroids)
        labels = jnp.argmin(distances, axis=1).astype(jnp.int32)
        sums, counts = _cluster_sums(transformed, labels, n_clusters)
        means = _means_from_sums(sums, counts)
        order = _farthest_order(transformed, labels, means)
        centroids = _fill_empty(transformed, means, counts, order)
    return labels


def qrkc(
    queries,
    keys,
    values,
    n_clusters: int,
    n_iters: int,
    alpha: float = DEFAULT_ALPHA,
    eps: float = DEFAULT_EPS,
) -> QRKCResult:
    """Cluster keys in the query-conditioned metric and reorder K and V.

    G_Q is the shrunk query metric, R is its Cholesky factor with G_Q = R R^T,
    and the transformed keys are K @ R. Euclidean Lloyd clustering runs on those
    keys only. Labels are then applied to the original keys and values.
    Permutation is argsort(labels). Centroids are means in the original key
    space, not the transformed space.

    Empty clusters are re-seeded from the farthest transformed key so centroids
    stay finite. Initialization is deterministic farthest-point selection. The
    paper only says to initialize centroids from the transformed keys.
    """
    queries = jnp.asarray(queries)
    keys = jnp.asarray(keys)
    values = jnp.asarray(values)
    if queries.ndim != 2 or keys.ndim != 2 or values.ndim != 2:
        raise ValueError("queries, keys, and values must be rank-2")
    if queries.shape[0] != keys.shape[0] or values.shape[0] != keys.shape[0]:
        raise ValueError("queries, keys, and values must have the same token count")
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("queries and keys must share the head dimension")
    n_clusters = int(n_clusters)
    n_iters = int(n_iters)
    if n_clusters < 1 or n_clusters > keys.shape[0]:
        raise ValueError(
            f"n_clusters must be in 1..L, got {n_clusters} for L={keys.shape[0]}"
        )
    if n_iters < 1:
        raise ValueError(f"n_iters must be at least 1, got {n_iters}")

    metric = shrunk_query_metric(queries, alpha=alpha, eps=eps)
    factor = jnp.linalg.cholesky(metric)
    transformed = keys @ factor
    labels = _lloyd(transformed, n_clusters, n_iters)
    permutation = jnp.argsort(labels)

    sums, counts = _cluster_sums(keys, labels, n_clusters)
    means = _means_from_sums(sums, counts)
    transformed_sums, _ = _cluster_sums(transformed, labels, n_clusters)
    transformed_means = _means_from_sums(transformed_sums, counts)
    order = _farthest_order(transformed, labels, transformed_means)
    centroids = _fill_empty(keys, means, counts, order)

    return QRKCResult(
        labels=labels,
        permutation=permutation,
        keys=keys[permutation],
        values=values[permutation],
        centroids=centroids,
        sizes=counts,
        cholesky_factor=factor,
        metric=metric,
        transformed_keys=transformed,
    )
