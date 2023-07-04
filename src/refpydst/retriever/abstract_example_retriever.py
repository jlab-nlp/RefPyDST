"""
This class defines an abstract example retriever, which can be used to retriever demonstrations from a training set for
a prompting experiment. These can include dense and sparse retrievers, oracles, etc.
"""
import abc
from typing import List

from refpydst.data_types import Turn

from refpydst.retriever.abstract_example_set_decoder import AbstractExampleListDecoder


class ExampleRetriever(abc.ABC):

    @abc.abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def item_to_best_examples(self, turn: Turn, k: int = 10, decoder: AbstractExampleListDecoder = None) \
            -> List[Turn]:
        """
        Given a target turn, return the k nearest turns in the training set according to this retriever

        :param turn:
        :param k:
        :param decoder:
        :return:
        """
        pass


class MockRetriever(ExampleRetriever):
    def __init__(self, **kwargs) -> None:
        pass

    def item_to_best_examples(self, turn: Turn, k: int = 10, decoder: AbstractExampleListDecoder = None) -> \
    List[Turn]:
        raise NotImplementedError("misconfigured experiment: not expecting this method to be called!")