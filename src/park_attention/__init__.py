"""JAX kernel for PARK block retrieval (arXiv:2609.38978)."""

from park_attention.attention import attention_recall, dense_attention, sparse_attention
from park_attention.pama import centroid_average_importance, pama
from park_attention.qrkc import (
    DEFAULT_ALPHA,
    DEFAULT_EPS,
    S_MIN,
    QRKCResult,
    qrkc,
    shrunk_query_metric,
)
from park_attention.retrieval import exact_block_importance, token_mask, top_p_block_mask
from park_attention.synthetic import disagreement_case

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_EPS",
    "QRKCResult",
    "S_MIN",
    "attention_recall",
    "centroid_average_importance",
    "dense_attention",
    "disagreement_case",
    "exact_block_importance",
    "pama",
    "qrkc",
    "shrunk_query_metric",
    "sparse_attention",
    "token_mask",
    "top_p_block_mask",
]

__version__ = "0.1.0"
