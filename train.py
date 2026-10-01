"""Print PAMA and centroid-average recall on the synthetic disagreement case."""

from park_attention import (
    attention_recall,
    centroid_average_importance,
    dense_attention,
    disagreement_case,
    pama,
    token_mask,
    top_p_block_mask,
)


def main() -> None:
    case = disagreement_case()
    threshold = 0.9
    pama_importance = pama(
        case.queries, case.key_centroids, case.key_sizes, case.query_block_ids
    )
    centroid_importance = centroid_average_importance(
        case.queries, case.key_centroids, case.key_sizes, case.query_block_ids
    )
    _, attention = dense_attention(case.queries, case.keys, case.values)
    pama_mask = token_mask(
        top_p_block_mask(pama_importance, threshold),
        case.query_block_ids,
        case.key_block_ids,
    )
    centroid_mask = token_mask(
        top_p_block_mask(centroid_importance, threshold),
        case.query_block_ids,
        case.key_block_ids,
    )
    pama_recall = float(attention_recall(attention, pama_mask))
    centroid_recall = float(attention_recall(attention, centroid_mask))
    print(f"pama_recall {pama_recall:.6f} centroid_recall {centroid_recall:.6f}")


if __name__ == "__main__":
    main()
