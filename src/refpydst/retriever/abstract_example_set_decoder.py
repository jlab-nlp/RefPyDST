import abc
from collections import Iterator

from refpydst.data_types import Turn


class AbstractExampleListDecoder(abc.ABC):
    """
    Example set decoders address to decoding portion of example selection as a structured prediction problem. Given an
    iterator over possible individual examples and their local scores, in descending order, according to a dense
    retrieval model such as an EmbeddingRetriever, the decoder selects the best K examples to form a demonstration set,
    in some order.

    The simplest and default decoder is to just select the 10 highest scoring examples in ascending order (such that
    when put in a prompt, the example nearest to the inference turn is highest scoring).
    """

    @abc.abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def select_k(self, examples: Iterator[Turn, float], k: int):
        pass
