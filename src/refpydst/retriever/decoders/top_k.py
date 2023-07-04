from typing import Iterator, Tuple, List

from refpydst.data_types import Turn

from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder


class TopKDecoder(AbstractExampleListDecoder):
    """
    An example selection decoder which simply takes the 10 highest scoring examples.
    """
    def __init__(self, **kwargs) -> None:
        pass

    def select_k(self, examples: Iterator[Tuple[Turn, float]], k: int) -> List[Turn]:
        # first 10 in iterator are highest scoring, reverse so highest is last
        return [turn for _, (turn, score) in zip(range(k), examples)][::-1]