from typing import Tuple, Iterator, List

import numpy as np
from numpy.typing import NDArray
from refpydst.data_types import Turn
from sklearn.metrics.pairwise import cosine_similarity

from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder
from refpydst.retriever.code.embed_based_retriever import EmbeddingRetriever


class MaximizeEmbeddingDistinctness(AbstractExampleListDecoder):
    """
    An example selection decoder which simply takes the 10 highest scoring examples.
    """
    from_n_possible: int
    discount_factor: float
    retriever: EmbeddingRetriever

    def __init__(self, retriever: EmbeddingRetriever, from_n_possible: int, discount_factor: float = 0.1, **kwargs) -> None:
        self.from_n_possible = from_n_possible
        self.discount_factor = discount_factor
        self.retriever = retriever

    def select_k(self, examples: Iterator[Tuple[Turn, float]], k: int) -> List[Turn]:
        # Select up to "from N possible" in the iterator
        all_considered_examples: List[Tuple[Turn, float]] = \
            [turn_and_score for _, turn_and_score in zip(range(self.from_n_possible), examples)]
        # shape: (example_idx, embedding size)
        all_embeddings: NDArray = np.asarray([
            self.retriever.label_to_search_embedding(self.retriever.data_item_to_label(turn))
            for turn, score in all_considered_examples
        ])
        if len(all_considered_examples) == 0:
            return []
        # storing these as a list of indices so they can be interpreted: 0-(k-1) would mean our discount factor had no
        # impact on the score.
        result: List[int] = []

        # scores from retriever are euclidean distances, of unit vectors (range 0-2). cos(y, z) = 1-.5*euc(y, z)^2
        # We initialize example_scores with the cosine similarity of each example e to x, the input turn (implicit from
        #  order of argument examples to select_k).
        example_scores: NDArray = np.asarray([1 - 0.5*(score**2) for turn, score in all_considered_examples])
        assert np.all(np.diff(example_scores) <= 0)  # verifies they are decreasing as expected
        while len(result) < k:
            # find the current best scoring example
            best_idx: int = np.argmax(example_scores).item()
            example_scores[best_idx] = -np.inf
            result.append(best_idx)
            # Update the scores. The worst-case decrease in score is defined by discount_factor.
            best_emb: NDArray = all_embeddings[best_idx]
            discount: NDArray = self.discount_factor * cosine_similarity(best_emb[None, :], all_embeddings).squeeze(0)
            example_scores = example_scores - discount
        return [all_considered_examples[i][0] for i in result][::-1]
